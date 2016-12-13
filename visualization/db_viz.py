#: Plotting tool to visualize data from arbitrary lift
#: Pulls directly from database
from sqlalchemy import create_engine
from pandas import read_sql
from sqlalchemy.exc import OperationalError

from databasing.db_conn_strings import aws_conn_string
from databasing.database_pull import pull_data_by_lift

conn = create_engine(aws_conn_string)


query = '''
SELECT
    al.lift_id,
    al.lift_type,
    ai.athlete_last_name || ', ' || ai.athlete_first_name AS athlete_name,
    ai.athlete_id
FROM athlete_lift AS al
INNER JOIN athlete_info AS ai
    ON al.athlete_id = ai.athlete_id
ORDER BY al.lift_type, al.lift_id DESC
'''

available_lifts = read_sql(query, conn)

lift_id = available_lifts['lift_id'].ix[0]
header, data = pull_data_by_lift(lift_id)

from bokeh.plotting import figure, output_file, show

# Arbitrary filename that Bokeh will then load up into default browser from local
output_file('test.html')

TOOLS = 'hover,box_zoom,box_select,reset,crosshair,pan'
# Any figure specifications go here
p = figure(plot_width=1000, plot_height=600, title='Test Plot: Lift {}'.format(lift_id), toolbar_location='right', tools=TOOLS)

# simple
# p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
p.line(data['timepoint'], data['a_x'], line_width=1)

# Forces bokeh to load up plot in browser
show(p)

