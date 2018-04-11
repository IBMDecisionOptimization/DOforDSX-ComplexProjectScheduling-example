from docplex.cp.model import *
from docplex.cp.expression import _FLOATING_POINT_PRECISION
import time

import pandas as pd
import numpy as np


schedUnitPerDurationUnit = 1  # DurationUnit is seconds
duration_units_per_day = 86400.0

# Define global constants for date to integer conversions
horizon_start_date = pd.to_datetime('Sat Apr 23 08:00:00 UTC 2016')
horizon_end_date = pd.to_datetime('Tue Apr 21 08:00:00 UTC 2026')
nanosecs_per_sec = 1000.0 * 1000 * 1000
secs_per_day = 3600.0 * 24

# Calendars handling
all_calendar_intervals_by_keys = dict()
all_calendar_intervals_by_keys['default'] = pd.DataFrame(columns=['start', 'end'])

internal_calendar_col_id = 'internal_calendar'
default_calendar_id = 'default'


def helper_create_internal_calendar_column(target_df):
    if not internal_calendar_col_id in target_df.columns:
        target_df[internal_calendar_col_id] = default_calendar_id

# Parse and convert a date Series to an integer Series
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_date_series_to_int(date_series):
    return (pd.to_numeric((date_series - horizon_start_date).values) / nanosecs_per_sec * duration_units_per_day * schedUnitPerDurationUnit / secs_per_day).astype(np.int64)

# Convert type to 'int64'
def helper_int64_convert(arg):
    if pd.__version__ < '0.20.0':
        return arg.astype('int64', raise_on_error=False)
    else:
        return arg.astype('int64', errors='ignore')

# Parse and convert an integer Series to a date Series
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_int_series_to_date(sched_int_series):
    return pd.to_datetime(sched_int_series * secs_per_day / duration_units_per_day / schedUnitPerDurationUnit * nanosecs_per_sec + horizon_start_date.value, errors='coerce')

def helper_update_interval_calendars(main_target_df, filtered_target_df, id_col, new_calendar_id, new_calendar_intervals_df):
    flat_target_df = main_target_df.reset_index()
    update_key_ids = len(main_target_df[main_target_df.internal_calendar.isin(filtered_target_df.internal_calendar.unique())]) != len(filtered_target_df)
    grpby = filtered_target_df.reset_index()[[id_col, internal_calendar_col_id]].groupby(internal_calendar_col_id)
    for k, v in grpby:
        current_calendar_intervals = all_calendar_intervals_by_keys[k]
        new_key = k + '+' + new_calendar_id if update_key_ids or k == default_calendar_id else k
        new_calendar_intervals = current_calendar_intervals.append(new_calendar_intervals_df, ignore_index=True)
        all_calendar_intervals_by_keys[new_key] = new_calendar_intervals
        if update_key_ids or k == default_calendar_id:
            flat_target_df.loc[flat_target_df[id_col].isin(v[id_col]), internal_calendar_col_id] = new_key
    main_target_df[internal_calendar_col_id] = flat_target_df.set_index(main_target_df.index.names)[internal_calendar_col_id]

# Convert a duration Series to a Series representing the number of scheduling units
def helper_convert_duration_series_to_scheduling_unit(duration_series, nb_input_data_units_per_day):
    return helper_int64_convert(duration_series * duration_units_per_day * schedUnitPerDurationUnit / nb_input_data_units_per_day)

# Parse and convert a date to an integer
# Integer value represents the number of schedule units (time granularity for engine) since horizon start
def helper_convert_date_to_int(date):
    return int((date - horizon_start_date).value / nanosecs_per_sec * duration_units_per_day * schedUnitPerDurationUnit / secs_per_day)


def helper_parse_and_convert_date_to_int(date_as_str):
    return helper_convert_date_to_int(pd.to_datetime(date_as_str))

# Label constraint
expr_counter = 1
def helper_add_labeled_cpo_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(expr_counter)
        expr_counter += 1
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)

from_to_break_calendar_intervals_by_keys = dict()


def helper_get_or_create_from_to_break_calendar(date_from, date_to):
    key = 'break_calendar_' + str(date_from) + '_' + str(date_to)
    from_to_break_calendar = from_to_break_calendar_intervals_by_keys.get(key, None)
    if from_to_break_calendar is not None:
        return key, from_to_break_calendar
    try:
        from_to_break_calendar = pd.DataFrame([[date_from, date_to]], columns=['start', 'end'])
        from_to_break_calendar_intervals_by_keys[key] = from_to_break_calendar
        return key, from_to_break_calendar
    except ValueError:
        return 'default', all_calendar_intervals_by_keys['default']  # return default calendar

# Create default calendar
def helper_get_default_calendar():
    calendar = CpoStepFunction()
    calendar.set_value(-INTERVAL_MAX, INTERVAL_MAX, 100)
    return calendar


# Create all calendars (step functions) and assign them to their respective tasks
def helper_build_all_break_calendars():
    all_break_calendars_by_keys = dict()
    for k in all_calendar_intervals_by_keys.keys():
        # Create calendar will 100% availability over planning horizon, then add break intervals
        calendar = helper_get_default_calendar()
        for interval in all_calendar_intervals_by_keys[k].itertuples(index=False):
            calendar.set_value(int(interval.start), int(interval.end), 0)
        all_break_calendars_by_keys[k] = calendar
    return all_break_calendars_by_keys



# Data model definition for each table
# Data collection: list_of_Calendar ['Id']
# Data collection: list_of_Date_Resource_Availability ['Available_Capacity', 'Calendar', 'End_Date', 'Start_Date']
# Data collection: list_of_FCResource_Request ['Finite_Capacity_Resource', 'Quantity', 'Task']
# Data collection: list_of_Finite_Capacity_Resource ['Capacity', 'Calendar', 'Id']
# Data collection: list_of_Individual_Resource ['Pool', 'Calendar', 'Id']
# Data collection: list_of_Module ['Id']
# Data collection: list_of_Modules_Synchronization ['Description', 'Origin_Module', 'Synchronization_Type', 'Target_Module']
# Data collection: list_of_Pool ['Id']
# Data collection: list_of_Pool_Resource_Request ['Pool', 'Quantity', 'Skill', 'Task']
# Data collection: list_of_Resource_Skill ['Skill', 'Worker', 'line']
# Data collection: list_of_Skill ['Id']
# Data collection: list_of_Synchronization_Type ['Id']
# Data collection: list_of_Task ['Transition_State', 'Breaks_Calendar', 'Fixed_Duration_Sec', 'Id', 'Is_Optional', 'Module', 'Unperformed_Penalty_Cost']
# Data collection: list_of_Task_Module_Synchronization ['Origin_Task', 'Synchronization_Type', 'Target_Module']
# Data collection: list_of_Tasks_Synchronization ['Description', 'Origin_Task', 'Synchronization_Type', 'Target_Task']
# Data collection: list_of_Transition_Matrix ['From', 'To', 'Transition_time_sec', 'Transition_cost', 'line']
# Data collection: list_of_Transition_State ['Id']

# Create a pandas Dataframe for each data table
list_of_Calendar = inputs['Calendar']
list_of_Calendar = list_of_Calendar[['Id']].copy()
list_of_Calendar.rename(columns={'Id': 'Id'}, inplace=True)
list_of_Date_Resource_Availability = inputs['Date_Resource_Availability']
list_of_Date_Resource_Availability = list_of_Date_Resource_Availability[['Available Capacity', 'Calendar', 'End Date', 'Start Date']].copy()
list_of_Date_Resource_Availability.rename(columns={'Available Capacity': 'Available_Capacity', 'Calendar': 'Calendar', 'End Date': 'End_Date', 'Start Date': 'Start_Date'}, inplace=True)
list_of_Date_Resource_Availability['End_Date'] = pd.to_datetime(list_of_Date_Resource_Availability['End_Date'])
list_of_Date_Resource_Availability['Start_Date'] = pd.to_datetime(list_of_Date_Resource_Availability['Start_Date'])
list_of_FCResource_Request = inputs['FCResource_Request']
list_of_FCResource_Request = list_of_FCResource_Request[['Finite Capacity Resource', 'Quantity', 'Task']].copy()
list_of_FCResource_Request.rename(columns={'Finite Capacity Resource': 'Finite_Capacity_Resource', 'Quantity': 'Quantity', 'Task': 'Task'}, inplace=True)
list_of_Finite_Capacity_Resource = inputs['Finite_Capacity_Resource']
list_of_Finite_Capacity_Resource = list_of_Finite_Capacity_Resource[['Capacity', 'Calendar', 'Id']].copy()
list_of_Finite_Capacity_Resource.rename(columns={'Capacity': 'Capacity', 'Calendar': 'Calendar', 'Id': 'Id'}, inplace=True)
list_of_Individual_Resource = inputs['Individual_Resource']
list_of_Individual_Resource = list_of_Individual_Resource[['Pool', 'Calendar', 'Id']].copy()
list_of_Individual_Resource.rename(columns={'Pool': 'Pool', 'Calendar': 'Calendar', 'Id': 'Id'}, inplace=True)
list_of_Module = inputs['Module']
list_of_Module = list_of_Module[['Id']].copy()
list_of_Module.rename(columns={'Id': 'Id'}, inplace=True)
list_of_Modules_Synchronization = inputs['Modules_Synchronization']
list_of_Modules_Synchronization = list_of_Modules_Synchronization[['Description', 'Origin Module', 'Synchronization Type', 'Target Module']].copy()
list_of_Modules_Synchronization.rename(columns={'Description': 'Description', 'Origin Module': 'Origin_Module', 'Synchronization Type': 'Synchronization_Type', 'Target Module': 'Target_Module'}, inplace=True)
list_of_Pool = inputs['Pool']
list_of_Pool = list_of_Pool[['Id']].copy()
list_of_Pool.rename(columns={'Id': 'Id'}, inplace=True)
list_of_Pool_Resource_Request = inputs['Pool_Resource_Request']
list_of_Pool_Resource_Request = list_of_Pool_Resource_Request[['Pool', 'Quantity', 'Skill', 'Task']].copy()
list_of_Pool_Resource_Request.rename(columns={'Pool': 'Pool', 'Quantity': 'Quantity', 'Skill': 'Skill', 'Task': 'Task'}, inplace=True)
list_of_Resource_Skill = inputs['Resource_Skill']
list_of_Resource_Skill = list_of_Resource_Skill[['Skill', 'Worker']].copy()
list_of_Resource_Skill.rename(columns={'Skill': 'Skill', 'Worker': 'Worker'}, inplace=True)
list_of_Skill = inputs['Skill']
list_of_Skill = list_of_Skill[['Id']].copy()
list_of_Skill.rename(columns={'Id': 'Id'}, inplace=True)
list_of_Synchronization_Type = inputs['Synchronization_Type']
list_of_Synchronization_Type = list_of_Synchronization_Type[['Id']].copy()
list_of_Synchronization_Type.rename(columns={'Id': 'Id'}, inplace=True)
list_of_Task = inputs['Task']
list_of_Task = list_of_Task[['Transition State', 'Breaks Calendar', 'Fixed Duration Sec', 'Id', 'Is Optional', 'Module', 'Unperformed Penalty Cost']].copy()
list_of_Task.rename(columns={'Transition State': 'Transition_State', 'Breaks Calendar': 'Breaks_Calendar', 'Fixed Duration Sec': 'Fixed_Duration_Sec', 'Id': 'Id', 'Is Optional': 'Is_Optional', 'Module': 'Module', 'Unperformed Penalty Cost': 'Unperformed_Penalty_Cost'}, inplace=True)
list_of_Task_Module_Synchronization = inputs['Task_Module_Synchronization']
list_of_Task_Module_Synchronization = list_of_Task_Module_Synchronization[['Origin Task', 'Synchronization Type', 'Target Module']].copy()
list_of_Task_Module_Synchronization.rename(columns={'Origin Task': 'Origin_Task', 'Synchronization Type': 'Synchronization_Type', 'Target Module': 'Target_Module'}, inplace=True)
list_of_Tasks_Synchronization = inputs['Tasks_Synchronization']
list_of_Tasks_Synchronization = list_of_Tasks_Synchronization[['Description', 'Origin Task', 'Synchronization Type', 'Target Task']].copy()
list_of_Tasks_Synchronization.rename(columns={'Description': 'Description', 'Origin Task': 'Origin_Task', 'Synchronization Type': 'Synchronization_Type', 'Target Task': 'Target_Task'}, inplace=True)
list_of_Transition_Matrix = inputs['Transition_Matrix']
list_of_Transition_Matrix = list_of_Transition_Matrix[['From', 'To', 'Transition time sec', 'Transition cost']].copy()
list_of_Transition_Matrix.rename(columns={'From': 'From', 'To': 'To', 'Transition time sec': 'Transition_time_sec', 'Transition cost': 'Transition_cost'}, inplace=True)
list_of_Transition_State = inputs['Transition_State']
list_of_Transition_State = list_of_Transition_State[['Id']].copy()
list_of_Transition_State.rename(columns={'Id': 'Id'}, inplace=True)

# Convert all input dates to integer to be used by optimizer engine API
list_of_Date_Resource_Availability['End_Date'] = helper_convert_date_series_to_int(list_of_Date_Resource_Availability.End_Date)
list_of_Date_Resource_Availability['Start_Date'] = helper_convert_date_series_to_int(list_of_Date_Resource_Availability.Start_Date)
# Convert all input durations to internal time unit
list_of_Task['Fixed_Duration_Sec'] = helper_convert_duration_series_to_scheduling_unit(list_of_Task.Fixed_Duration_Sec, 86400.0)
list_of_Transition_Matrix['Transition_time_sec'] = helper_convert_duration_series_to_scheduling_unit(list_of_Transition_Matrix.Transition_time_sec, 86400.0)

# Set index when a primary key is defined
list_of_Calendar.set_index('Id', inplace=True)
list_of_Calendar.sort_index(inplace=True)
list_of_Calendar.index.name = 'id_of_Calendar'
list_of_Date_Resource_Availability.set_index('Start_Date', inplace=True)
list_of_Date_Resource_Availability.sort_index(inplace=True)
list_of_Date_Resource_Availability.index.name = 'id_of_Date_Resource_Availability'
list_of_FCResource_Request.set_index('Task', inplace=True)
list_of_FCResource_Request.sort_index(inplace=True)
list_of_FCResource_Request.index.name = 'id_of_Task'
list_of_Finite_Capacity_Resource.set_index('Id', inplace=True)
list_of_Finite_Capacity_Resource.sort_index(inplace=True)
list_of_Finite_Capacity_Resource.index.name = 'id_of_Finite_Capacity_Resource'
list_of_Individual_Resource.set_index('Id', inplace=True)
list_of_Individual_Resource.sort_index(inplace=True)
list_of_Individual_Resource.index.name = 'id_of_Individual_Resource'
list_of_Module.set_index('Id', inplace=True)
list_of_Module.sort_index(inplace=True)
list_of_Module.index.name = 'id_of_Module'
list_of_Modules_Synchronization.set_index('Description', inplace=True)
list_of_Modules_Synchronization.sort_index(inplace=True)
list_of_Modules_Synchronization.index.name = 'id_of_Modules_Synchronization'
list_of_Pool.set_index('Id', inplace=True)
list_of_Pool.sort_index(inplace=True)
list_of_Pool.index.name = 'id_of_Pool'
list_of_Pool_Resource_Request.set_index('Task', inplace=True)
list_of_Pool_Resource_Request.sort_index(inplace=True)
list_of_Pool_Resource_Request.index.name = 'id_of_Task'
list_of_Resource_Skill.index.name = 'id_of_Resource_Skill'
list_of_Skill.set_index('Id', inplace=True)
list_of_Skill.sort_index(inplace=True)
list_of_Skill.index.name = 'id_of_Skill'
list_of_Synchronization_Type.set_index('Id', inplace=True)
list_of_Synchronization_Type.sort_index(inplace=True)
list_of_Synchronization_Type.index.name = 'id_of_Synchronization_Type'
list_of_Task.set_index('Id', inplace=True)
list_of_Task.sort_index(inplace=True)
list_of_Task.index.name = 'id_of_Task'
list_of_Task_Module_Synchronization.set_index('Target_Module', inplace=True)
list_of_Task_Module_Synchronization.sort_index(inplace=True)
list_of_Task_Module_Synchronization.index.name = 'id_of_Module'
list_of_Tasks_Synchronization.set_index('Description', inplace=True)
list_of_Tasks_Synchronization.sort_index(inplace=True)
list_of_Tasks_Synchronization.index.name = 'id_of_Tasks_Synchronization'
list_of_Transition_Matrix.index.name = 'id_of_Transition_Matrix'
list_of_Transition_State.set_index('Id', inplace=True)
list_of_Transition_State.sort_index(inplace=True)
list_of_Transition_State.index.name = 'id_of_Module'

# Create data frame as cartesian product of: Task x Individual_Resource
list_of_SchedulingAssignment = pd.DataFrame(index=pd.MultiIndex.from_product((list_of_Task.index, list_of_Individual_Resource.index), names=['id_of_Task', 'id_of_Individual_Resource']))


def build_model():
    mdl = CpoModel()

    # Definition of model variables
    list_of_SchedulingAssignment['interval'] = interval_var_list(len(list_of_SchedulingAssignment), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_SchedulingAssignment['schedulingAssignmentVar'] = list_of_SchedulingAssignment.interval.apply(mdl.presence_of)
    list_of_Task['interval'] = interval_var_list(len(list_of_Task), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_Task['taskStartVar'] = list_of_Task.interval.apply(mdl.start_of)
    list_of_Task['taskEndVar'] = list_of_Task.interval.apply(mdl.end_of)
    list_of_Task['taskDurationVar'] = list_of_Task.interval.apply(mdl.size_of)
    list_of_SchedulingAssignment['taskAssignmentDurationVar'] = list_of_SchedulingAssignment.interval.apply(mdl.size_of)
    list_of_Task['taskAbsenceVar'] = 1 - list_of_Task.interval.apply(mdl.presence_of)
    list_of_Task['taskPresenceVar'] = list_of_Task.interval.apply(mdl.presence_of)
    list_of_Module['interval'] = interval_var_list(len(list_of_Module), end=(INTERVAL_MIN, INTERVAL_MAX / 4), optional=True)
    list_of_Module['taskStartVar'] = list_of_Module.interval.apply(mdl.start_of)
    list_of_Module['taskEndVar'] = list_of_Module.interval.apply(mdl.end_of)
    list_of_Module['taskDurationVar'] = list_of_Module.interval.apply(mdl.size_of)
    list_of_Module['taskAbsenceVar'] = 1 - list_of_Module.interval.apply(mdl.presence_of)
    list_of_Module['taskPresenceVar'] = list_of_Module.interval.apply(mdl.presence_of)
    
    # Compute an index in range 0..n for each transition type, to be used to index entries in transition matrix
    Transition_State_transition_id_by_index_df = pd.DataFrame(list_of_Task.Transition_State.unique(), index=range(1, len(list_of_Task.Transition_State.unique()) + 1), columns=['transition_id'])
    Transition_State_transition_index_by_id_df = Transition_State_transition_id_by_index_df.reset_index().set_index('transition_id')
    list_of_Task['Transition_State_index'] = list_of_Task.join(Transition_State_transition_index_by_id_df, on='Transition_State')['index']
    # Build list of intervals and interval types to be used in 'sequence_vars' definition for Unary Resources
    join_SchedulingAssignment_Task = list_of_SchedulingAssignment.join(list_of_Task.Transition_State_index, how='inner')
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Individual_Resource.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.interval.groupby(level=groupbyLevels).apply(list).to_frame(name='interval')
    groupbyLevels = [join_SchedulingAssignment_Task.index.names.index(name) for name in list_of_Individual_Resource.index.names]
    groupby_SchedulingAssignment_Task = join_SchedulingAssignment_Task.Transition_State_index.groupby(level=groupbyLevels).apply(list).to_frame(name='Transition_State_index')
    # Create a sequence variable for each Unary Resource
    list_of_Individual_Resource['sequence_var'] = groupby_SchedulingAssignment.join(groupby_SchedulingAssignment_Task).apply(lambda row: sequence_var(row.interval, row.Transition_State_index), axis=1)
    
    # For each FCResource_Request, Task requires Quantity of Finite Capacity Resource
    join_FCResource_Request_Task = list_of_FCResource_Request.join(list_of_Task.interval, how='inner')
    join_FCResource_Request_Task['usage'] = [pulse(row.interval, row.Quantity) for row in join_FCResource_Request_Task.itertuples(index=False)]
    list_of_Finite_Capacity_Resource_usage = join_FCResource_Request_Task[['Finite_Capacity_Resource', 'usage']].groupby('Finite_Capacity_Resource').sum()


    # Definition of model
    # Objective (Guided) Minimize time to complete all task-
    # Combine weighted criteria: 
    # 	cMinimizeMakespan cMinimizeMakespan{
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cMinimizeMakespan.taskEnd = cTaskEnd[Task],
    # 	cSingleCriterionGoal.numericExpr = max of count( cTaskEnd[Task]) over cTaskEnd[Task],
    # 	cMinimizeMakespan.task = Task} with weight 5.0
    # 	cMinimizeGoalScheduling cMinimizeGoalScheduling{
    # 	cScaledGoal.scaleFactorExpr = 1,
    # 	cSingleCriterionGoal.goalFilter = null,
    # 	cSingleCriterionGoal.numericExpr = total cTaskAbsence[Task] / Task / Unperformed Penalty Cost} with weight 5.0
    # 	cExplicitMinimizeTransitionCostGoal cExplicitMinimizeTransitionCostGoal{
    # 	cExplicitMinimizeTransitionCostGoal.task = Task,
    # 	cExplicitMinimizeTransitionCostGoal.transitionMatrix = Transition_Matrix,
    # 	cExplicitMinimizeTransitionCostGoal.assignment = cSchedulingAssignment[Task, Individual_Resource]} with weight 5.0
    agg_Task_taskEndVar_SG1 = mdl.max(list_of_Task.taskEndVar)
    list_of_Task['conditioned_Unperformed_Penalty_Cost'] = list_of_Task.taskAbsenceVar * list_of_Task.Unperformed_Penalty_Cost
    agg_Task_conditioned_Unperformed_Penalty_Cost_SG2 = mdl.sum(list_of_Task.conditioned_Unperformed_Penalty_Cost)
    # For each transition type 'origin', build actual transition values, indexed by transition type 'index'
    transition_cost_from_id_to_index = list_of_Transition_Matrix.reset_index().join(Transition_State_transition_index_by_id_df, on='To', rsuffix='_to')
    transition_cost_from_id_to_index.columns = list(list_of_Transition_Matrix.reset_index()) + ['index_to']
    transition_cost_from_id_to_index.set_index(['From', 'index_to'], inplace=True)
    transition_cost_from_id_to_index.sort_index(inplace=True)
    transition_cost_from_id_to_index = transition_cost_from_id_to_index.Transition_cost.groupby(level=0).apply(lambda r: np.append([0], list(r))).to_frame(name='Transition_cost')
    # Build expression for each Scheduling Assignment for evaluating associated transition cost
    join_SchedulingAssignment_Task_SG3 = list_of_SchedulingAssignment.join(list_of_Task.Transition_State, how='inner')
    join_SchedulingAssignment_Task_Individual_Resource_SG3 = join_SchedulingAssignment_Task_SG3.join(list_of_Individual_Resource.sequence_var, how='inner')
    join_SchedulingAssignment_Task_Individual_Resource_SG3 = join_SchedulingAssignment_Task_Individual_Resource_SG3.reset_index().join(transition_cost_from_id_to_index, on='Transition_State').set_index(join_SchedulingAssignment_Task_Individual_Resource_SG3.index.names)
    join_SchedulingAssignment_Task_Individual_Resource_SG3['idNextIndex'] = join_SchedulingAssignment_Task_Individual_Resource_SG3.apply(lambda row : type_of_next(row.sequence_var, row.interval, lastValue=0, absentValue=0), axis=1)
    join_SchedulingAssignment_Task_Individual_Resource_SG3['costNextIndex'] = join_SchedulingAssignment_Task_Individual_Resource_SG3.apply(lambda row : element(row.Transition_cost, row.idNextIndex), axis=1)
    agg_SchedulingAssignment_Task_Individual_Resource_SG3 = mdl.sum(join_SchedulingAssignment_Task_Individual_Resource_SG3.costNextIndex)
    
    kpi_1 = integer_var(name='kpi_1')
    mdl.add(kpi_1 >= 16.0 * (agg_Task_taskEndVar_SG1 / schedUnitPerDurationUnit) / 1 - 1 + _FLOATING_POINT_PRECISION)
    mdl.add(kpi_1 <= 16.0 * (agg_Task_taskEndVar_SG1 / schedUnitPerDurationUnit) / 1)
    mdl.add_kpi(kpi_1, name='time to complete all Tasks')
    kpi_2 = integer_var(name='kpi_2')
    mdl.add(kpi_2 >= 16.0 * (agg_Task_conditioned_Unperformed_Penalty_Cost_SG2) / 1 - 1 + _FLOATING_POINT_PRECISION)
    mdl.add(kpi_2 <= 16.0 * (agg_Task_conditioned_Unperformed_Penalty_Cost_SG2) / 1)
    mdl.add_kpi(kpi_2, name='total Unperformed Penalty Cost of unperformed Tasks')
    kpi_3 = integer_var(name='kpi_3')
    mdl.add(kpi_3 >= 16.0 * (agg_SchedulingAssignment_Task_Individual_Resource_SG3) / 1 - 1 + _FLOATING_POINT_PRECISION)
    mdl.add(kpi_3 <= 16.0 * (agg_SchedulingAssignment_Task_Individual_Resource_SG3) / 1)
    mdl.add_kpi(kpi_3, name='transit cost between assigned Tasks based on Transition_Matrices')
    
    mdl.add(minimize( 0
        # Sub Goal (Guided) Minimize time to complete all task_cMinimizeGoal
        # Minimize time to complete all Tasks
        + 16.0 * (agg_Task_taskEndVar_SG1 / schedUnitPerDurationUnit) / 1
        # Sub Goal (Guided) Minimize total Unperformed Penalty Cost of task of unperformed task_cMinimizeGoal
        # Minimize total Unperformed Penalty Cost of unperformed Tasks
        + 16.0 * (agg_Task_conditioned_Unperformed_Penalty_Cost_SG2) / 1
        # Sub Goal (Guided) Minimize transit cost between task of scheduling assignment based on Transition_Matrix_cBaseMinimizeTransitionGoal
        # Minimize transit cost between assigned Tasks based on Transition_Matrices
        + 16.0 * (agg_SchedulingAssignment_Task_Individual_Resource_SG3) / 1
    ))
    
    # [ST_1] Constraint : (Guided) All task where is optional of task is equal to 0 are present_cIterativeRelationalConstraint
    # All Tasks where Is Optional is equal to 0 are present
    # Label: CT_1_All_Tasks_where_Is_Optional_is_equal_to_0_are_present
    filtered_Task = list_of_Task[list_of_Task.Is_Optional == 0].copy()
    for row in filtered_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskAbsenceVar != 1, 'All Tasks where Is Optional is equal to 0 are present', row)
    
    # [ST_2] Constraint : (Guided) for each task , scheduled duration of task is equal to fixed duration sec of task_cIterativeRelationalConstraint
    # For each Task, scheduled duration is equal to Fixed Duration Sec
    # Label: CT_2_For_each_Task__scheduled_duration_is_equal_to_Fixed_Duration_Sec
    for row in list_of_Task[list_of_Task.Fixed_Duration_Sec.notnull()].itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, size_of(row.interval, int(row.Fixed_Duration_Sec)) == int(row.Fixed_Duration_Sec), 'For each Task, scheduled duration is equal to Fixed Duration Sec', row)
    
    # [ST_3] Constraint : (Guided) For each task , module of task is parent of task and spans over all children_cTaskSpanConstraint
    # For each Task, Module is parent of Task and spans over all children
    # Label: CT_3_For_each_Task__Module_is_parent_of_Task_and_spans_over_all_children
    join_Task_Module = list_of_Task.reset_index().join(list_of_Module.interval, on=['Module'], rsuffix='_right', how='inner').set_index(list_of_Task.index.names)
    groupby_Task_Module = join_Task_Module.reset_index()[['Module', 'interval']].groupby(['Module'])['interval'].apply(list).to_frame()
    join_Module_Task_Module = list_of_Module.join(groupby_Task_Module.interval, rsuffix='_right', how='inner')
    for row in join_Module_Task_Module.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, span(row.interval, row.interval_right), 'For each Task, Module is parent of Task and spans over all children', row)
    
    # [ST_4] Constraint : (Guided) For each Pool_Resource_Request, task of Pool_Resource_Request requires quantity of Pool_Resource_Request of pool of Pool_Resource_Request (grouped by pool of Individual_Resource)_cUnaryResourcePoolRequestConstraint
    # For each Pool_Resource_Request, Task requires Quantity of Pool (grouped by Pool of Individual_Resources)
    # Label: CT_4_For_each_Pool_Resource_Request__Task_requires_Quantity_of_Pool__grouped_by_Pool_of_Individual_Resources_
    join_SchedulingAssignment_Individual_Resource = list_of_SchedulingAssignment.join(list_of_Individual_Resource.Pool, how='inner')
    join_Pool_Resource_Request_Task = list_of_Pool_Resource_Request.join(list_of_Task.interval, how='inner')
    join_Pool_Resource_Request_SchedulingAssignment_Individual_Resource = list_of_Pool_Resource_Request.reset_index().merge(join_SchedulingAssignment_Individual_Resource.reset_index(), left_on=['id_of_Task', 'Pool'], right_on=['id_of_Task', 'Pool']).set_index(list_of_Pool_Resource_Request.index.names + list(set(join_SchedulingAssignment_Individual_Resource.index.names) - set(list_of_Pool_Resource_Request.index.names)))
    groupby_Pool_Resource_Request_SchedulingAssignment_Individual_Resource = join_Pool_Resource_Request_SchedulingAssignment_Individual_Resource.reset_index()[['id_of_Task', 'Pool', 'interval']].groupby(['id_of_Task', 'Pool'])['interval'].apply(list).to_frame()
    join_Pool_Resource_Request_SchedulingAssignment_Individual_Resource_Pool_Resource_Request_Task = groupby_Pool_Resource_Request_SchedulingAssignment_Individual_Resource.join(join_Pool_Resource_Request_Task, rsuffix='_right', how='inner')
    for row in join_Pool_Resource_Request_SchedulingAssignment_Individual_Resource_Pool_Resource_Request_Task[join_Pool_Resource_Request_SchedulingAssignment_Individual_Resource_Pool_Resource_Request_Task.Quantity.notnull()].itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, alternative(row.interval_right, row.interval, row.Quantity), 'For each Pool_Resource_Request, Task requires Quantity of Pool (grouped by Pool of Individual_Resources)', row)
    # No assignment of pools with no requests
    join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request = join_SchedulingAssignment_Individual_Resource.join(list_of_Pool_Resource_Request.Pool, rsuffix='_right')
    join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request = join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request.reset_index()
    join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request = join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request[join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request.Pool != join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request.Pool_right]
    for row in join_SchedulingAssignment_Individual_Resource_Pool_Resource_Request.itertuples(index=False):
        helper_add_labeled_cpo_constraint(mdl, logical_not(presence_of(row.interval)), 'For each Pool_Resource_Request, Task requires Quantity of Pool (grouped by Pool of Individual_Resources)')
    
    # [ST_5] Constraint : (Guided) For each scheduling assignment, skill of Resource_Skills of Individual_Resource of scheduling assignment includes Skill of Pool_Resource_Request of task of scheduling assignment_cCategoryCompatibilityConstraintOnPair
    # For each Individual_Resource to Task assignment, Skill of Resource_Skills of assigned Individual_Resources includes Skill of Pool_Resource_Requests of assigned Tasks
    # Label: CT_5_For_each_Individual_Resource_to_Task_assignment__Skill_of_Resource_Skills_of_assigned_Individual_Resources_includes_Skill_of_Pool_Resource_Requests_of_assigned_Tasks
    join_SchedulingAssignment_Resource_Skill = list_of_SchedulingAssignment.reset_index().merge(list_of_Resource_Skill.reset_index(), left_on=['id_of_Individual_Resource'], right_on=['Worker']).set_index(list_of_SchedulingAssignment.index.names + list(set(list_of_Resource_Skill.index.names) - set(list_of_SchedulingAssignment.index.names)))
    reindexed_SchedulingAssignment_Resource_Skill = join_SchedulingAssignment_Resource_Skill.reset_index().set_index(['id_of_Task', 'id_of_Individual_Resource'])
    join_SchedulingAssignment_Pool_Resource_Request = list_of_SchedulingAssignment.join(list_of_Pool_Resource_Request.Skill, how='inner')
    join_SchedulingAssignment_Resource_Skill_SchedulingAssignment_Pool_Resource_Request = reindexed_SchedulingAssignment_Resource_Skill.join(join_SchedulingAssignment_Pool_Resource_Request.Skill, rsuffix='_right', how='inner')
    filtered_SchedulingAssignment_Resource_Skill_SchedulingAssignment_Pool_Resource_Request = join_SchedulingAssignment_Resource_Skill_SchedulingAssignment_Pool_Resource_Request[join_SchedulingAssignment_Resource_Skill_SchedulingAssignment_Pool_Resource_Request.Skill == join_SchedulingAssignment_Resource_Skill_SchedulingAssignment_Pool_Resource_Request.Skill_right].copy()
    helper_add_labeled_cpo_constraint(mdl, mdl.sum(join_SchedulingAssignment_Pool_Resource_Request.schedulingAssignmentVar[(join_SchedulingAssignment_Pool_Resource_Request.Skill.notnull()) & (~join_SchedulingAssignment_Pool_Resource_Request.index.isin(filtered_SchedulingAssignment_Resource_Skill_SchedulingAssignment_Pool_Resource_Request.index.values))]) == 0, 'For each Individual_Resource to Task assignment, Skill of Resource_Skills of assigned Individual_Resources includes Skill of Pool_Resource_Requests of assigned Tasks')
    
    # [ST_6] Constraint : (Guided) For each Tasks_Synchronization where Synchronization Type of Tasks_Synchronization is Synchronization_Type : StartsAfterEnd, Target Task of Tasks_Synchronization starts after the end of Origin Task of Tasks_Synchronization_cTaskPredecessorsConstraint
    # For each Tasks_Synchronization where Synchronization Type is StartsAfterEnd, Target Task starts after the end of Origin Task
    # Label: CT_6_For_each_Tasks_Synchronization_where_Synchronization_Type_is_StartsAfterEnd__Target_Task_starts_after_the_end_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAfterEnd']]
    filtered_Tasks_Synchronization = list_of_Tasks_Synchronization[list_of_Tasks_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Tasks_Synchronization_Task = list_of_Tasks_Synchronization.reset_index().join(list_of_Task.interval, on=['Target_Task'], how='inner').set_index(list_of_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Tasks_Synchronization_Task = filtered_Tasks_Synchronization.join(join_Tasks_Synchronization_Task.interval, how='inner')
    join_Tasks_Synchronization_Task = list_of_Tasks_Synchronization.reset_index().join(list_of_Task.interval, on=['Origin_Task'], how='inner').set_index(list_of_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Tasks_Synchronization_Task_Tasks_Synchronization_Task = join_Tasks_Synchronization_Tasks_Synchronization_Task.join(join_Tasks_Synchronization_Task.interval, rsuffix='_right', how='inner')
    for row in join_Tasks_Synchronization_Tasks_Synchronization_Task_Tasks_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, end_before_start(row.interval_right, row.interval), 'For each Tasks_Synchronization where Synchronization Type is StartsAfterEnd, Target Task starts after the end of Origin Task', row)
    
    # [ST_7] Constraint : (Guided) For each Tasks_Synchronization where Synchronization Type of Tasks_Synchronization is Synchronization_Type : StartsAfterStart, Target Task of Tasks_Synchronization starts after the start of Origin Task of Tasks_Synchronization_cTaskPredecessorsConstraint
    # For each Tasks_Synchronization where Synchronization Type is StartsAfterStart, Target Task starts after the start of Origin Task
    # Label: CT_7_For_each_Tasks_Synchronization_where_Synchronization_Type_is_StartsAfterStart__Target_Task_starts_after_the_start_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAfterStart']]
    filtered_Tasks_Synchronization = list_of_Tasks_Synchronization[list_of_Tasks_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Tasks_Synchronization_Task = list_of_Tasks_Synchronization.reset_index().join(list_of_Task.interval, on=['Target_Task'], how='inner').set_index(list_of_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Tasks_Synchronization_Task = filtered_Tasks_Synchronization.join(join_Tasks_Synchronization_Task.interval, how='inner')
    join_Tasks_Synchronization_Task = list_of_Tasks_Synchronization.reset_index().join(list_of_Task.interval, on=['Origin_Task'], how='inner').set_index(list_of_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Tasks_Synchronization_Task_Tasks_Synchronization_Task = join_Tasks_Synchronization_Tasks_Synchronization_Task.join(join_Tasks_Synchronization_Task.interval, rsuffix='_right', how='inner')
    for row in join_Tasks_Synchronization_Tasks_Synchronization_Task_Tasks_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, start_before_start(row.interval_right, row.interval), 'For each Tasks_Synchronization where Synchronization Type is StartsAfterStart, Target Task starts after the start of Origin Task', row)
    
    # [ST_8] Constraint : (Guided) For each Tasks_Synchronization where Synchronization Type of Tasks_Synchronization is Synchronization_Type : StartsAtStart, scheduled start of Target Task of Tasks_Synchronization is equal to scheduled start of Origin Task of Tasks_Synchronization_cIterativeRelationalConstraint
    # For each Tasks_Synchronization where Synchronization Type is StartsAtStart, scheduled start of Target Task is equal to scheduled start of Origin Task
    # Label: CT_8_For_each_Tasks_Synchronization_where_Synchronization_Type_is_StartsAtStart__scheduled_start_of_Target_Task_is_equal_to_scheduled_start_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAtStart']]
    filtered_Tasks_Synchronization = list_of_Tasks_Synchronization[list_of_Tasks_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Tasks_Synchronization_Task = filtered_Tasks_Synchronization.reset_index().join(list_of_Task.taskStartVar, on=['Target_Task'], how='inner').set_index(filtered_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Task_2 = list_of_Tasks_Synchronization.reset_index().join(list_of_Task.taskStartVar, on=['Origin_Task'], how='inner').set_index(list_of_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Task_Tasks_Synchronization_Task = join_Tasks_Synchronization_Task.join(join_Tasks_Synchronization_Task_2.taskStartVar, rsuffix='_right', how='inner')
    for row in join_Tasks_Synchronization_Task_Tasks_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskStartVar == row.taskStartVar_right, 'For each Tasks_Synchronization where Synchronization Type is StartsAtStart, scheduled start of Target Task is equal to scheduled start of Origin Task', row)
    
    # [ST_9] Constraint : (Guided) For each Tasks_Synchronization where Synchronization Type of Tasks_Synchronization is Synchronization_Type : StartsAtEnd, scheduled start of Target Task of Tasks_Synchronization is equal to scheduled end of Origin Task of Tasks_Synchronization_cIterativeRelationalConstraint
    # For each Tasks_Synchronization where Synchronization Type is StartsAtEnd, scheduled start of Target Task is equal to scheduled end of Origin Task
    # Label: CT_9_For_each_Tasks_Synchronization_where_Synchronization_Type_is_StartsAtEnd__scheduled_start_of_Target_Task_is_equal_to_scheduled_end_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAtEnd']]
    filtered_Tasks_Synchronization = list_of_Tasks_Synchronization[list_of_Tasks_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Tasks_Synchronization_Task = filtered_Tasks_Synchronization.reset_index().join(list_of_Task.taskStartVar, on=['Target_Task'], how='inner').set_index(filtered_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Task_2 = list_of_Tasks_Synchronization.reset_index().join(list_of_Task.taskEndVar, on=['Origin_Task'], how='inner').set_index(list_of_Tasks_Synchronization.index.names)
    join_Tasks_Synchronization_Task_Tasks_Synchronization_Task = join_Tasks_Synchronization_Task.join(join_Tasks_Synchronization_Task_2.taskEndVar, how='inner')
    for row in join_Tasks_Synchronization_Task_Tasks_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskStartVar == row.taskEndVar, 'For each Tasks_Synchronization where Synchronization Type is StartsAtEnd, scheduled start of Target Task is equal to scheduled end of Origin Task', row)
    
    # [ST_10] Constraint : (Guided) For each Modules_Synchronization where Synchronization Type of Modules_Synchronization is Synchronization_Type : StartsAfterEnd, Target Module of Modules_Synchronization starts after the end of Origin Module of Modules_Synchronization_cTaskPredecessorsConstraint
    # For each Modules_Synchronization where Synchronization Type is StartsAfterEnd, Target Module starts after the end of Origin Module
    # Label: CT_10_For_each_Modules_Synchronization_where_Synchronization_Type_is_StartsAfterEnd__Target_Module_starts_after_the_end_of_Origin_Module
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAfterEnd']]
    filtered_Modules_Synchronization = list_of_Modules_Synchronization[list_of_Modules_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Modules_Synchronization_Module = list_of_Modules_Synchronization.reset_index().join(list_of_Module.interval, on=['Target_Module'], how='inner').set_index(list_of_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Modules_Synchronization_Module = filtered_Modules_Synchronization.join(join_Modules_Synchronization_Module.interval, how='inner')
    join_Modules_Synchronization_Module = list_of_Modules_Synchronization.reset_index().join(list_of_Module.interval, on=['Origin_Module'], how='inner').set_index(list_of_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Modules_Synchronization_Module_Modules_Synchronization_Module = join_Modules_Synchronization_Modules_Synchronization_Module.join(join_Modules_Synchronization_Module.interval, rsuffix='_right', how='inner')
    for row in join_Modules_Synchronization_Modules_Synchronization_Module_Modules_Synchronization_Module.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, end_before_start(row.interval_right, row.interval), 'For each Modules_Synchronization where Synchronization Type is StartsAfterEnd, Target Module starts after the end of Origin Module', row)
    
    # [ST_11] Constraint : (Guided) For each Modules_Synchronization where Synchronization Type of Modules_Synchronization is Synchronization_Type : StartsAfterStart, Target Module of Modules_Synchronization starts after the start of Origin Module of Modules_Synchronization_cTaskPredecessorsConstraint
    # For each Modules_Synchronization where Synchronization Type is StartsAfterStart, Target Module starts after the start of Origin Module
    # Label: CT_11_For_each_Modules_Synchronization_where_Synchronization_Type_is_StartsAfterStart__Target_Module_starts_after_the_start_of_Origin_Module
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAfterStart']]
    filtered_Modules_Synchronization = list_of_Modules_Synchronization[list_of_Modules_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Modules_Synchronization_Module = list_of_Modules_Synchronization.reset_index().join(list_of_Module.interval, on=['Target_Module'], how='inner').set_index(list_of_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Modules_Synchronization_Module = filtered_Modules_Synchronization.join(join_Modules_Synchronization_Module.interval, how='inner')
    join_Modules_Synchronization_Module = list_of_Modules_Synchronization.reset_index().join(list_of_Module.interval, on=['Origin_Module'], how='inner').set_index(list_of_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Modules_Synchronization_Module_Modules_Synchronization_Module = join_Modules_Synchronization_Modules_Synchronization_Module.join(join_Modules_Synchronization_Module.interval, rsuffix='_right', how='inner')
    for row in join_Modules_Synchronization_Modules_Synchronization_Module_Modules_Synchronization_Module.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, start_before_start(row.interval_right, row.interval), 'For each Modules_Synchronization where Synchronization Type is StartsAfterStart, Target Module starts after the start of Origin Module', row)
    
    # [ST_12] Constraint : (Guided) For each Modules_Synchronization where Synchronization Type of Modules_Synchronization is Synchronization_Type : StartsAtStart, scheduled start of Target Module of Modules_Synchronization is equal to scheduled start of Origin Module of Modules_Synchronization_cIterativeRelationalConstraint
    # For each Modules_Synchronization where Synchronization Type is StartsAtStart, scheduled start of Target Module is equal to scheduled start of Origin Module
    # Label: CT_12_For_each_Modules_Synchronization_where_Synchronization_Type_is_StartsAtStart__scheduled_start_of_Target_Module_is_equal_to_scheduled_start_of_Origin_Module
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAtStart']]
    filtered_Modules_Synchronization = list_of_Modules_Synchronization[list_of_Modules_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Modules_Synchronization_Module = filtered_Modules_Synchronization.reset_index().join(list_of_Module.taskStartVar, on=['Target_Module'], how='inner').set_index(filtered_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Module_2 = list_of_Modules_Synchronization.reset_index().join(list_of_Module.taskStartVar, on=['Origin_Module'], how='inner').set_index(list_of_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Module_Modules_Synchronization_Module = join_Modules_Synchronization_Module.join(join_Modules_Synchronization_Module_2.taskStartVar, rsuffix='_right', how='inner')
    for row in join_Modules_Synchronization_Module_Modules_Synchronization_Module.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskStartVar == row.taskStartVar_right, 'For each Modules_Synchronization where Synchronization Type is StartsAtStart, scheduled start of Target Module is equal to scheduled start of Origin Module', row)
    
    # [ST_13] Constraint : (Guided) For each Modules_Synchronization where Synchronization Type of Modules_Synchronization is Synchronization_Type : StartsAtEnd, scheduled start of Target Module of Modules_Synchronization is equal to scheduled end of Origin Module of Modules_Synchronization_cIterativeRelationalConstraint
    # For each Modules_Synchronization where Synchronization Type is StartsAtEnd, scheduled start of Target Module is equal to scheduled end of Origin Module
    # Label: CT_13_For_each_Modules_Synchronization_where_Synchronization_Type_is_StartsAtEnd__scheduled_start_of_Target_Module_is_equal_to_scheduled_end_of_Origin_Module
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAtEnd']]
    filtered_Modules_Synchronization = list_of_Modules_Synchronization[list_of_Modules_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Modules_Synchronization_Module = filtered_Modules_Synchronization.reset_index().join(list_of_Module.taskStartVar, on=['Target_Module'], how='inner').set_index(filtered_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Module_2 = list_of_Modules_Synchronization.reset_index().join(list_of_Module.taskEndVar, on=['Origin_Module'], how='inner').set_index(list_of_Modules_Synchronization.index.names)
    join_Modules_Synchronization_Module_Modules_Synchronization_Module = join_Modules_Synchronization_Module.join(join_Modules_Synchronization_Module_2.taskEndVar, how='inner')
    for row in join_Modules_Synchronization_Module_Modules_Synchronization_Module.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskStartVar == row.taskEndVar, 'For each Modules_Synchronization where Synchronization Type is StartsAtEnd, scheduled start of Target Module is equal to scheduled end of Origin Module', row)
    
    # [ST_14] Constraint : (Guided) For each Task_Module_Synchronization where Synchronization Type of Task_Module_Synchronization is Synchronization_Type : StartsAfterEnd, Target Module of Task_Module_Synchronization starts after the end of Origin Task of Task_Module_Synchronization_cTaskPredecessorsConstraint
    # For each Task_Module_Synchronization where Synchronization Type is StartsAfterEnd, Target Module starts after the end of Origin Task
    # Label: CT_14_For_each_Task_Module_Synchronization_where_Synchronization_Type_is_StartsAfterEnd__Target_Module_starts_after_the_end_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAfterEnd']]
    filtered_Task_Module_Synchronization = list_of_Task_Module_Synchronization[list_of_Task_Module_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Task_Module_Synchronization_Module = list_of_Task_Module_Synchronization.join(list_of_Module.interval, how='inner')
    join_Task_Module_Synchronization_Task_Module_Synchronization_Module = filtered_Task_Module_Synchronization.join(join_Task_Module_Synchronization_Module.interval, how='inner')
    join_Task_Module_Synchronization_Task = list_of_Task_Module_Synchronization.reset_index().join(list_of_Task.interval, on=['Origin_Task'], how='inner').set_index(list_of_Task_Module_Synchronization.index.names)
    join_Task_Module_Synchronization_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task = join_Task_Module_Synchronization_Task_Module_Synchronization_Module.join(join_Task_Module_Synchronization_Task.interval, rsuffix='_right', how='inner')
    for row in join_Task_Module_Synchronization_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, end_before_start(row.interval_right, row.interval), 'For each Task_Module_Synchronization where Synchronization Type is StartsAfterEnd, Target Module starts after the end of Origin Task', row)
    
    # [ST_15] Constraint : (Guided) For each Task_Module_Synchronization where Synchronization Type of Task_Module_Synchronization is Synchronization_Type : StartsAfterStart, Target Module of Task_Module_Synchronization starts after the start of Origin Task of Task_Module_Synchronization_cTaskPredecessorsConstraint
    # For each Task_Module_Synchronization where Synchronization Type is StartsAfterStart, Target Module starts after the start of Origin Task
    # Label: CT_15_For_each_Task_Module_Synchronization_where_Synchronization_Type_is_StartsAfterStart__Target_Module_starts_after_the_start_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAfterStart']]
    filtered_Task_Module_Synchronization = list_of_Task_Module_Synchronization[list_of_Task_Module_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Task_Module_Synchronization_Module = list_of_Task_Module_Synchronization.join(list_of_Module.interval, how='inner')
    join_Task_Module_Synchronization_Task_Module_Synchronization_Module = filtered_Task_Module_Synchronization.join(join_Task_Module_Synchronization_Module.interval, how='inner')
    join_Task_Module_Synchronization_Task = list_of_Task_Module_Synchronization.reset_index().join(list_of_Task.interval, on=['Origin_Task'], how='inner').set_index(list_of_Task_Module_Synchronization.index.names)
    join_Task_Module_Synchronization_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task = join_Task_Module_Synchronization_Task_Module_Synchronization_Module.join(join_Task_Module_Synchronization_Task.interval, rsuffix='_right', how='inner')
    for row in join_Task_Module_Synchronization_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, start_before_start(row.interval_right, row.interval), 'For each Task_Module_Synchronization where Synchronization Type is StartsAfterStart, Target Module starts after the start of Origin Task', row)
    
    # [ST_16] Constraint : (Guided) For each Task_Module_Synchronization where Synchronization Type of Task_Module_Synchronization is Synchronization_Type : StartsAtStart, scheduled start of Target Module of Task_Module_Synchronization is equal to scheduled start of Origin Task of Task_Module_Synchronization_cIterativeRelationalConstraint
    # For each Task_Module_Synchronization where Synchronization Type is StartsAtStart, scheduled start of Target Module is equal to scheduled start of Origin Task
    # Label: CT_16_For_each_Task_Module_Synchronization_where_Synchronization_Type_is_StartsAtStart__scheduled_start_of_Target_Module_is_equal_to_scheduled_start_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAtStart']]
    filtered_Task_Module_Synchronization = list_of_Task_Module_Synchronization[list_of_Task_Module_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Task_Module_Synchronization_Module = filtered_Task_Module_Synchronization.join(list_of_Module.taskStartVar, how='inner')
    join_Task_Module_Synchronization_Task = list_of_Task_Module_Synchronization.reset_index().join(list_of_Task.taskStartVar, on=['Origin_Task'], how='inner').set_index(list_of_Task_Module_Synchronization.index.names)
    join_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task = join_Task_Module_Synchronization_Module.join(join_Task_Module_Synchronization_Task.taskStartVar, rsuffix='_right', how='inner')
    for row in join_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskStartVar == row.taskStartVar_right, 'For each Task_Module_Synchronization where Synchronization Type is StartsAtStart, scheduled start of Target Module is equal to scheduled start of Origin Task', row)
    
    # [ST_17] Constraint : (Guided) For each Task_Module_Synchronization where Synchronization Type of Task_Module_Synchronization is Synchronization_Type : StartsAtEnd, scheduled start of Target Module of Task_Module_Synchronization is equal to scheduled end of Origin Task of Task_Module_Synchronization_cIterativeRelationalConstraint
    # For each Task_Module_Synchronization where Synchronization Type is StartsAtEnd, scheduled start of Target Module is equal to scheduled end of Origin Task
    # Label: CT_17_For_each_Task_Module_Synchronization_where_Synchronization_Type_is_StartsAtEnd__scheduled_start_of_Target_Module_is_equal_to_scheduled_end_of_Origin_Task
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['StartsAtEnd']]
    filtered_Task_Module_Synchronization = list_of_Task_Module_Synchronization[list_of_Task_Module_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Task_Module_Synchronization_Module = filtered_Task_Module_Synchronization.join(list_of_Module.taskStartVar, how='inner')
    join_Task_Module_Synchronization_Task = list_of_Task_Module_Synchronization.reset_index().join(list_of_Task.taskEndVar, on=['Origin_Task'], how='inner').set_index(list_of_Task_Module_Synchronization.index.names)
    join_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task = join_Task_Module_Synchronization_Module.join(join_Task_Module_Synchronization_Task.taskEndVar, how='inner')
    for row in join_Task_Module_Synchronization_Module_Task_Module_Synchronization_Task.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.taskStartVar == row.taskEndVar, 'For each Task_Module_Synchronization where Synchronization Type is StartsAtEnd, scheduled start of Target Module is equal to scheduled end of Origin Task', row)
    
    # [ST_18] Constraint : (Guided) For each Task_Module_Synchronization where Synchronization Type of Task_Module_Synchronization is Synchronization_Type : EndsBeforeStart, Origin Task of Task_Module_Synchronization starts after the end of Target Module of Task_Module_Synchronization_cTaskPredecessorsConstraint
    # For each Task_Module_Synchronization where Synchronization Type is EndsBeforeStart, Origin Task starts after the end of Target Module
    # Label: CT_18_For_each_Task_Module_Synchronization_where_Synchronization_Type_is_EndsBeforeStart__Origin_Task_starts_after_the_end_of_Target_Module
    filtered_Synchronization_Type = list_of_Synchronization_Type.loc[['EndsBeforeStart']]
    filtered_Task_Module_Synchronization = list_of_Task_Module_Synchronization[list_of_Task_Module_Synchronization.Synchronization_Type.isin(filtered_Synchronization_Type.index)]
    join_Task_Module_Synchronization_Task = list_of_Task_Module_Synchronization.reset_index().join(list_of_Task.interval, on=['Origin_Task'], how='inner').set_index(list_of_Task_Module_Synchronization.index.names)
    join_Task_Module_Synchronization_Task_Module_Synchronization_Task = filtered_Task_Module_Synchronization.join(join_Task_Module_Synchronization_Task.interval, how='inner')
    join_Task_Module_Synchronization_Module = list_of_Task_Module_Synchronization.join(list_of_Module.interval, how='inner')
    join_Task_Module_Synchronization_Task_Module_Synchronization_Task_Task_Module_Synchronization_Module = join_Task_Module_Synchronization_Task_Module_Synchronization_Task.join(join_Task_Module_Synchronization_Module.interval, rsuffix='_right', how='inner')
    for row in join_Task_Module_Synchronization_Task_Module_Synchronization_Task_Task_Module_Synchronization_Module.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, end_before_start(row.interval_right, row.interval), 'For each Task_Module_Synchronization where Synchronization Type is EndsBeforeStart, Origin Task starts after the end of Target Module', row)
    
    # [ST_19] Constraint : (Guided) The schedule must comply with the capacity limit defined for each Finite_Capacity_Resource_cDiscreteResourceCapacityLimitConstraint
    # The schedule must comply with the capacity limit defined for each Finite_Capacity_Resource
    # Label: CT_19_The_schedule_must_comply_with_the_capacity_limit_defined_for_each_Finite_Capacity_Resource
    join_list_of_Finite_Capacity_Resource_usage = list_of_Finite_Capacity_Resource_usage.join(list_of_Finite_Capacity_Resource, how='inner')
    for row in join_list_of_Finite_Capacity_Resource_usage.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, row.usage <= int(row.Capacity), 'The schedule must comply with the capacity limit defined for each Finite_Capacity_Resource', row)
    
    # [ST_20] Constraint : (Guided) for each Date_Resource_Availability, the capacity of Finite_Capacity_Resources of Calendar of Date_Resource_Availability is limited to Available Capacity of Date_Resource_Availability between Start Date of Date_Resource_Availability and End Date of Date_Resource_Availability_cDiscreteResourceCapacityLimitConstraint
    # For each Date_Resource_Availability, the capacity of Finite_Capacity_Resources of Calendar is limited to Available Capacity between Start Date and End Date
    # Label: CT_20_For_each_Date_Resource_Availability__the_capacity_of_Finite_Capacity_Resources_of_Calendar_is_limited_to_Available_Capacity_between_Start_Date_and_End_Date
    join_Date_Resource_Availability_Calendar = list_of_Date_Resource_Availability.reset_index().join(list_of_Calendar, on=['Calendar'], how='inner').set_index(list_of_Date_Resource_Availability.index.names)
    join_Date_Resource_Availability_Calendar_Finite_Capacity_Resource = join_Date_Resource_Availability_Calendar.reset_index().merge(list_of_Finite_Capacity_Resource.reset_index(), left_on=['Calendar'], right_on=['Calendar']).set_index(join_Date_Resource_Availability_Calendar.index.names + list(set(list_of_Finite_Capacity_Resource.index.names) - set(join_Date_Resource_Availability_Calendar.index.names)))
    join_list_of_Finite_Capacity_Resource_usage = join_Date_Resource_Availability_Calendar_Finite_Capacity_Resource.reset_index().join(list_of_Finite_Capacity_Resource_usage.usage, on='id_of_Finite_Capacity_Resource', how='inner')
    for row in join_list_of_Finite_Capacity_Resource_usage.itertuples(index=True):
        helper_add_labeled_cpo_constraint(mdl, always_in(row.usage, (int(row.id_of_Date_Resource_Availability), int(row.End_Date)), 0, int(row.Available_Capacity)), 'For each Date_Resource_Availability, the capacity of Finite_Capacity_Resources of Calendar is limited to Available Capacity between Start Date and End Date', row)
    
    # [ST_21] Constraint : (Guided) For each Date_Resource_Availability, add an unavailable period for Individual_Resources of Calendar of Date_Resource_Availability between Start Date of Date_Resource_Availability and End Date of Date_Resource_Availability_cUnaryResourceUnavailabilityConstraint
    # For each Date_Resource_Availability, add an unavailable period for Individual_Resources of Calendar between Start Date and End Date
    # Label: CT_21_For_each_Date_Resource_Availability__add_an_unavailable_period_for_Individual_Resources_of_Calendar_between_Start_Date_and_End_Date
    join_Date_Resource_Availability_Calendar = list_of_Date_Resource_Availability.reset_index().join(list_of_Calendar, on=['Calendar'], how='inner').set_index(list_of_Date_Resource_Availability.index.names)
    join_Date_Resource_Availability_Calendar_Individual_Resource = join_Date_Resource_Availability_Calendar.reset_index().merge(list_of_Individual_Resource.reset_index(), left_on=['Calendar'], right_on=['Calendar']).set_index(join_Date_Resource_Availability_Calendar.index.names + list(set(list_of_Individual_Resource.index.names) - set(join_Date_Resource_Availability_Calendar.index.names)))
    reindexed_Date_Resource_Availability_Calendar_Individual_Resource = join_Date_Resource_Availability_Calendar_Individual_Resource.reset_index().set_index(['id_of_Date_Resource_Availability'])
    join_SchedulingAssignment_Date_Resource_Availability_Calendar_Individual_Resource = list_of_SchedulingAssignment.reset_index().merge(reindexed_Date_Resource_Availability_Calendar_Individual_Resource.reset_index(), left_on=['id_of_Individual_Resource'], right_on=['id_of_Individual_Resource']).set_index(list_of_SchedulingAssignment.index.names + list(set(reindexed_Date_Resource_Availability_Calendar_Individual_Resource.index.names) - set(list_of_SchedulingAssignment.index.names)))
    for row in join_SchedulingAssignment_Date_Resource_Availability_Calendar_Individual_Resource.reset_index().itertuples(index=False):
        helper_add_labeled_cpo_constraint(mdl, forbid_extent(row.interval, CpoStepFunction([(INTERVAL_MIN, 1), (int(row.id_of_Date_Resource_Availability), 0), (int(row.End_Date), 1)])), 'For each Date_Resource_Availability, add an unavailable period for Individual_Resources of Calendar between Start Date and End Date', row, ['id_of_Task', 'id_of_Individual_Resource', 'id_of_Date_Resource_Availability'])
    
    # [ST_22] Constraint : (Guided) For each Date_Resource_Availability, add a break on tasks of Calendar of Date_Resource_Availability between Start Date of Date_Resource_Availability and End Date of Date_Resource_Availability_cTaskDateBreakConstraintAbstract
    # For each Date_Resource_Availability, add a break on Tasks of Calendar between Start Date and End Date
    # Label: CT_22_For_each_Date_Resource_Availability__add_a_break_on_Tasks_of_Calendar_between_Start_Date_and_End_Date
    helper_create_internal_calendar_column(list_of_Task)
    join_Date_Resource_Availability_Calendar = list_of_Date_Resource_Availability.reset_index().join(list_of_Calendar, on=['Calendar'], how='inner').set_index(list_of_Date_Resource_Availability.index.names)
    join_Date_Resource_Availability_Calendar_Task = join_Date_Resource_Availability_Calendar.reset_index().merge(list_of_Task.reset_index(), left_on=['Calendar'], right_on=['Breaks_Calendar']).set_index(join_Date_Resource_Availability_Calendar.index.names + list(set(list_of_Task.index.names) - set(join_Date_Resource_Availability_Calendar.index.names)))
    reindexed_Date_Resource_Availability_Calendar_Task = join_Date_Resource_Availability_Calendar_Task.reset_index().set_index(['id_of_Date_Resource_Availability'])
    groupby_Date_Resource_Availability_Calendar_Task = reindexed_Date_Resource_Availability_Calendar_Task.reset_index()[['id_of_Date_Resource_Availability', 'End_Date', 'id_of_Task']].groupby(['id_of_Date_Resource_Availability', 'End_Date'])['id_of_Task'].apply(list).to_frame()
    for row in groupby_Date_Resource_Availability_Calendar_Task.reset_index().itertuples(index=False):
        calendar_id, day_break_intervals = helper_get_or_create_from_to_break_calendar(int(row.id_of_Date_Resource_Availability), int(row.End_Date))
        helper_update_interval_calendars(list_of_Task, list_of_Task[list_of_Task.index.isin(row.id_of_Task)], list_of_Task.index.name, calendar_id, day_break_intervals)
    
    # [ST_23] Constraint : (Guided) Apply a minimum transit time between task of scheduling assignment based on Transition_Matrix_cBaseMinimumTransitionTimeConstraint
    # Apply a minimum transit time between assigned Tasks based on Transition_Matrices
    # Label: CT_23_Apply_a_minimum_transit_time_between_assigned_Tasks_based_on_Transition_Matrices
    # For each transition type 'origin', build actual transition values, indexed by transition type 'index'
    transition_time_from_id_to_index = list_of_Transition_Matrix.reset_index().join(Transition_State_transition_index_by_id_df, on='From', rsuffix='_from')
    transition_time_from_id_to_index = transition_time_from_id_to_index.join(Transition_State_transition_index_by_id_df, on='To', rsuffix='_to')
    transition_time_from_id_to_index.columns = list(list_of_Transition_Matrix.reset_index()) + ['index_from', 'index_to']
    transition_time_from_id_to_index.set_index(['index_from', 'index_to'], inplace=True)
    transition_time_from_id_to_index.sort_index(inplace=True)
    transition_time_from_id_to_index = transition_time_from_id_to_index.Transition_time_sec.groupby(level=0).apply(lambda r: np.append([0], list(r)))
    transition_time_from_id_to_index = pd.Series([np.zeros(1 + len(transition_time_from_id_to_index), dtype=np.int64)]).append(transition_time_from_id_to_index)
    transitionTimeMatrix = transition_matrix(transition_time_from_id_to_index.values)
    
    # Configure all tasks with their respective calendar
    all_break_calendars_by_keys = helper_build_all_break_calendars()
    for row in list_of_Task[list_of_Task.internal_calendar != default_calendar_id].itertuples(index=False):
        row.interval.set_intensity(all_break_calendars_by_keys[row.internal_calendar])
        mdl.add(forbid_start(row.interval, all_break_calendars_by_keys[row.internal_calendar]))
        mdl.add(forbid_end(row.interval, all_break_calendars_by_keys[row.internal_calendar]))
    
    # link presence if not alternative
    groupbyLevels = [list_of_SchedulingAssignment.index.names.index(name) for name in list_of_Task.index.names]
    groupby_SchedulingAssignment = list_of_SchedulingAssignment.schedulingAssignmentVar.groupby(level=groupbyLevels).agg(lambda l: mdl.max(l.tolist())).to_frame()
    join_SchedulingAssignment_Task = groupby_SchedulingAssignment.join(list_of_Task.taskPresenceVar, how='inner')
    for row in join_SchedulingAssignment_Task.itertuples(index=False):
        mdl.add(row.schedulingAssignmentVar <= row.taskPresenceVar)
    
    # no overlap
    for row in list_of_Individual_Resource.reset_index().itertuples(index=False):
        mdl.add(no_overlap(row.sequence_var, transitionTimeMatrix))


    return mdl


def solve_model(mdl):
    params = CpoParameters()
    params.TimeLimit = 60
    solver = CpoSolver(mdl, params=params, trace_log=True)
    try:
        for i, msol in enumerate(solver):
            ovals = msol.get_objective_values()
            print("Objective values: {}".format(ovals))
            for k, v in msol.get_kpis().iteritems():
                print k, '-->', v
            export_solution(msol)
            if ovals is None:
                break  # No objective: stop after first solution
        # If model is infeasible, invoke conflict refiner to return
        if solver.get_last_solution().get_solve_status() == SOLVE_STATUS_INFEASIBLE:
            conflicts = solver.refine_conflict()
            export_conflicts(conflicts)
    except CpoException as e:
        # Solve has been aborted from an external action
        print('An exception has been raised: %s' % str(e))
        raise e


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print conflicts
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for item, index in zip(conflicts.member_constraints, range(len(conflicts.member_constraints))):
        label, context = expr_to_info.get(item.name, ('N/A', item.name))
        constraint_detail = expression._to_string(item)
        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, constraint_detail))
        list_of_conflicts = list_of_conflicts.append(pd.DataFrame({'constraint': label, 'context': str(context), 'detail': constraint_detail},
                                                                  index=[index], columns=['constraint', 'context', 'detail']))

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    list_of_SchedulingAssignment_solution = pd.DataFrame(index=list_of_SchedulingAssignment.index)
    list_of_SchedulingAssignment_solution['schedulingAssignmentVar'] = list_of_SchedulingAssignment.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Task_solution = pd.DataFrame(index=list_of_Task.index)
    list_of_Task_solution = list_of_Task_solution.join(pd.DataFrame([msol.solution[interval] if msol.solution[interval] else (None, None, None) for interval in list_of_Task.interval], index=list_of_Task.index, columns=['taskStartVar', 'taskEndVar', 'taskDurationVar']))
    list_of_Task_solution['taskStartVarDate'] = helper_convert_int_series_to_date(list_of_Task_solution.taskStartVar)
    list_of_Task_solution['taskEndVarDate'] = helper_convert_int_series_to_date(list_of_Task_solution.taskEndVar)
    list_of_Task_solution.taskStartVar /= schedUnitPerDurationUnit
    list_of_Task_solution.taskEndVar /= schedUnitPerDurationUnit
    list_of_Task_solution.taskDurationVar /= schedUnitPerDurationUnit
    list_of_SchedulingAssignment_solution['taskAssignmentDurationVar'] = list_of_SchedulingAssignment.interval.apply(lambda r: msol.solution.get_var_solution(r).get_size() if msol.solution.get_var_solution(r) else np.NaN)
    list_of_SchedulingAssignment_solution.taskAssignmentDurationVar /= schedUnitPerDurationUnit
    list_of_Task_solution['taskAbsenceVar'] = list_of_Task.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_absent() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Task_solution['taskPresenceVar'] = list_of_Task.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Module_solution = pd.DataFrame(index=list_of_Module.index)
    list_of_Module_solution = list_of_Module_solution.join(pd.DataFrame([msol.solution[interval] if msol.solution[interval] else (None, None, None) for interval in list_of_Module.interval], index=list_of_Module.index, columns=['taskStartVar', 'taskEndVar', 'taskDurationVar']))
    list_of_Module_solution['taskStartVarDate'] = helper_convert_int_series_to_date(list_of_Module_solution.taskStartVar)
    list_of_Module_solution['taskEndVarDate'] = helper_convert_int_series_to_date(list_of_Module_solution.taskEndVar)
    list_of_Module_solution.taskStartVar /= schedUnitPerDurationUnit
    list_of_Module_solution.taskEndVar /= schedUnitPerDurationUnit
    list_of_Module_solution.taskDurationVar /= schedUnitPerDurationUnit
    list_of_Module_solution['taskAbsenceVar'] = list_of_Module.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_absent() else 0) if msol.solution.get_var_solution(r) else np.NaN)
    list_of_Module_solution['taskPresenceVar'] = list_of_Module.interval.apply(lambda r: (1 if msol.solution.get_var_solution(r).is_present() else 0) if msol.solution.get_var_solution(r) else np.NaN)

    # Filter rows for non-selected assignments
    list_of_SchedulingAssignment_solution = list_of_SchedulingAssignment_solution[list_of_SchedulingAssignment_solution.schedulingAssignmentVar > 0.5]

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    with output_lock:
        outputs['list_of_Task_solution'] = list_of_Task_solution.reset_index()
        outputs['list_of_SchedulingAssignment_solution'] = list_of_SchedulingAssignment_solution.reset_index()
        outputs['list_of_Module_solution'] = list_of_Module_solution.reset_index()

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


print('* building wado model')
start_time = time.time()
model = build_model()
elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
solve_model(model)
elapsed_time = time.time() - start_time
print('solve + export of all intermediate solutions done in ' + str(elapsed_time) + ' secs')
