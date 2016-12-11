#: Plotting tool to visualize data from arbitrary lift
#: Pulls directly from database
from sqlalchemy import create_engine
from pandas import read_sql
from sqlalchemy.exc import OperationalError

# from databasing.db_conn_strings import aws_conn_string
# from databasing.database_pull import pull_data_by_lift
#
# conn = create_engine(aws_conn_string)
#
# data = pull_data_by_lift(1)


from bokeh.plotting import figure, output_file, show

# Arbitrary filename that Bokeh will then load up into default browser from local
output_file('test.html')

TOOLS = 'hover,box_zoom,box_select,reset,crosshair'
# Any figure specifications go here
p = figure(plot_width=600, plot_height=600, title='Test Plot', toolbar_location='right', tools=TOOLS)

# simple
p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
# p.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=10)

# Forces bokeh to load up plot in browser
show(p)
