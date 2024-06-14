import plotly.express as px
import pandas as pd

df = pd.DataFrame([
    dict(Task="Job A", Start='2009-01-01 08:00', Finish='2009-01-01 12:00', Name="Artur", Specialization="Programmer"),
    dict(Task="Job B", Start='2009-01-01 10:00', Finish='2009-01-01 14:00', Name="Weronika", Specialization="Programmer"),
    dict(Task="Job C", Start='2009-01-01 09:30', Finish='2009-01-01 16:00', Name="Paweł", Specialization="Programmer")
])

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Name", text="Task", labels="Specialization")
fig.update_yaxes(autorange="reversed")
fig.show()