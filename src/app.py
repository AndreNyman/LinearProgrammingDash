import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
import base64
import io
import json
import numpy as np

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1("Targeted marketing optimization",style={"color": "#1c3e4d", "font": "bold", 'margin': '20px'}),
    html.Br(),
    
    dcc.Markdown('''
    The targeted marketing or customer selection algorithm utilizes integer linear programming (ILP) to optimize a dual-optimization problem of maximizing generated revenue and customer retention, subject to the following constaints:
    
    1) the number of days since a customer was contacted > 14, 
    
    2) Marketing budget, 
    
    3) Total allowed number of customers and 
    
    4) Customer-allowed contacting channel 
                 
    ''', mathjax=True, style={'margin': '20px'}),

    dcc.Markdown('''
    PuLP is a modelling language that handles linear (LP), integer (IP) and mixed linear integer programming (MILP). The library can be connected to multiple solvers, such as CPLEX. This application uses the CBC solver to solve 
    
    the MILP problem described above. Solving linear models can roughly be divided into direct (e.g., decomposition methods) or iterative approaches, where the latter applies gradient or other information to iteratively search 
                 
    the optimal solution. PuLP utilizes an iterative approach, first relaxing the integer assumption to find the lower bound or the exact solution using the simplex or the interior-point approach. If the exact solution does not 
                 
    satisfy the integer constraint, then the method branches out on some variables in the model (branch and bound), fixing them to binary values and solving those subproblems. 
                 
    ''', mathjax=True, style={'margin': '20px'}),
    
    html.Br(),
    html.H3("Insert the customer data in the drag-and-drop field below",style={"color": "#1c3e4d", "font-weight": "bold",'margin': '20px'}),
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '20px'
        },
        multiple=False
        ),

        html.Div(id='uploaded-data', style={'margin': '20px'}),

        html.Br(),
        html.H3("Adjust the parameters below",style={"color": "#1c3e4d", "font-weight": "bold",'margin': '20px'}),
        html.Br(),
        #html.Label("Maximum Total Costs:"),
        dcc.Input(id='max-total-costs', type='number', placeholder = "Maximum Total Costs [eur]", style={'margin': '20px','width':'350px'}),
        html.Br(),
        #html.Label("Maximum Number of Customers to Target:"),
        dcc.Input(id='max-customers', type='number', placeholder = "Maximum Number of Customers to Target", style={'margin': '20px','width':'350px'}),
        html.Br(),
        #html.Label("Weight for Revenue Maximization:"),
        dcc.Input(id='weight-revenue', type='number', placeholder = "Weight for Revenue Maximization (0-1)", style={'margin': '20px','width':'350px'}),
        html.Br(),
        #html.Label("Weight for Retention Maximization:"),
        dcc.Input(id='weight-retention', type='number', placeholder = "Weight for Retention Maximization (0-1)", style={'margin': '20px','width':'350px'}),
        # dcc.Input(
        #     id="return_upper_bound", type="number", placeholder="Expected upper-bound return [%]",
        #     min=1, max=100, step=0.1,
        #     style={'width':'350px'}
        # ),
        html.Br(),

        html.Button('Run Optimization', id='run-optimization', style={'margin': '20px'}),
        html.Br(),
        html.Div(children=[
        html.Div(id="output_max_budget"),
        #html.Div(id="output_min_investment_per_Stock"),
        html.Div(id="output_max_nr_customers"),
        html.Div(id="output_weight_retention"),
        ], style={'padding': 1, 'flex': 1,'margin': '20px'}),
        html.Hr(),


        
        html.Br(),
        dcc.Store(id='optimization-results-result'),
        html.Br(),

        html.Div(id="output_obj_value1", style={'margin': '20px'}),
        #html.Br(),
        html.Div(children=[
        dash_table.DataTable(
            id='optimization-results-df',
            style_table={'overflowX': 'auto'},
            page_size=10,
            export_format='csv',  # Default export format
            export_headers='display',  # Include headers in the exported file
            merge_duplicate_headers=True,  # Merge duplicate headers if they exist
        )
        ], style={'padding': 1, 'flex': 1,'margin': '20px'}),
        
        
    ],
    fluid=True
)


def maximize_marketing_value_multi_objective(df, max_total_costs, max_customers, weight_revenue=1, weight_retention=1):
    prob = LpProblem("MaximizeMarketingValueMultiObjective", LpMaximize)

    customers = df.index.tolist()

    x = LpVariable.dicts("Customer", customers, cat='Binary')
    y = LpVariable.dicts("Channel", customers, cat='Binary')

    objective_revenue = lpSum(
        (df.loc[i, 'APV'] * df.loc[i, 'EngagementProbability'] - df.loc[i, 'AllowedChannelCost']) * y[i]
        for i in customers
    )

    objective_retention = lpSum(df.loc[i, 'ChurnRisk'] * y[i] for i in customers)
    if weight_revenue == None:
        weight_revenue = 1

    if weight_retention == None:
        weight_retention = 0
    prob += weight_revenue * objective_revenue + weight_retention * objective_retention

    prob += lpSum(df.loc[i, 'AllowedChannelCost'] * y[i] for i in customers) <= max_total_costs, "MaxTotalCostsConstraint"
    prob += lpSum(x[i] for i in customers) <= max_customers, "MaxCustomersConstraint"

    for i in customers:
        prob += x[i] == y[i], f"CustomerSelectionConstraint_{i}"
        prob += x[i] <= (1 if df.loc[i, 'DaysSinceContacted'] > 14 else 0), f"NotContactedLast14DaysConstraint_{i}"

    prob.solve()

    selected_customers = [i for i in customers if x[i].value() == 1]
    channel_costs = {i: df.loc[i, 'AllowedChannelCost'] for i in selected_customers}
    APV_col = {i: df.loc[i, 'APV'] for i in selected_customers}
    engagement_fractions = {i: df.loc[i, 'EngagementProbability'] for i in selected_customers}
    churnrisk_col = {i: df.loc[i, 'ChurnRisk'] for i in selected_customers}
    dayssincecontacted_col = {i: df.loc[i, 'DaysSinceContacted'] for i in selected_customers}
    allowedchannel_col = {i: df.loc[i, 'AllowedChannel'] for i in selected_customers}

    



    return {
        'ObjectiveRevenue': weight_revenue * objective_revenue.value(),
        'ObjectiveRetention': weight_retention * objective_retention.value(),
        'SelectedCustomers': selected_customers,
        'APV': APV_col,
        #'EngagementProbability': engagement_fractions,
        'EngagementFractions': engagement_fractions,  # <-- Include this line
        'ChurnRisk': churnrisk_col,
        'Channel': allowedchannel_col,
        'ChannelCosts': channel_costs,
        'DaysSinceContacted': dayssincecontacted_col
        
        
        
    }





def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    df = pd.read_excel(io.BytesIO(decoded), index_col='Customer')
    return df


def generate_selected_customers_data(df, optimization_results_df, optimization_results_result):
    result = json.loads(optimization_results_result)

    selected_customers_data = [
        {
            'Customer': customer,
            'Selected Channel': df.loc[customer, 'AllowedChannel'],
            'Channel Cost': result['ChannelCosts'][customer],
            'Engagement Fraction': result['EngagementFractions'][customer],
        }
        for customer in result['SelectedCustomers']
    ]

    return selected_customers_data


@app.callback(
    Output('output_max_budget', 'children'),
    #Output('output_min_investment_per_Stock', 'children'),
    Output('output_max_nr_customers', 'children'),
    #Output('output_weight_revenue', 'children'),
    Output('output_weight_retention', 'children'),
    Output('output_obj_value1', 'children'),
    [
        Input('run-optimization', 'n_clicks'),
    ],
    [
        State('max-total-costs', 'value'),
        State('max-customers', 'value'),
        State('weight-revenue', 'value'),
        State('weight-retention', 'value'),
        State('optimization-results-df','data')
    ]
)
def update_portfolios_scatterplot(n_clicks, max_budget, max_customers, weight_revenue, weight_retention, opt_results):
    
    if max_budget is not None:
        max_budget = 'The maximum allowed budget amount is: \n{}.'.format(max_budget)
    
    #if min_investment is not None:
    #    min_investment = 'The minimum required % investment per stock is \n{} %.'.format(min_investment)
    
    if max_customers is not None:
        max_customers = 'The upper limit on number of selected customers is set at \n{}.'.format(max_customers)
    
    #if weight_revenue is not None:
    #    weight_revenue = 'The lower limit on yearly expected returns is set at \n{} %.'.format(weight_revenue)
        
    if weight_revenue is not None:
        if weight_revenue >= 50:
            weight_revenue = 'The multi-objective-function problem is focusing \n{} % on profitability maximization.'.format(weight_revenue*100)
        else:
            weight_revenue = 'The multi-objective-function problem is focusing \n{} % on profitability maximization.'.format(weight_revenue*100)

    if opt_results is not None:
        opt_results = list(opt_results[0].values())[0]
        opt_results = 'The optimal objective value is:  \n{}.'.format(opt_results)
    
    return max_budget, max_customers, weight_revenue, opt_results



@app.callback(
    Output('uploaded-data', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def upload_file(contents, filename):
    if contents is None:
        return ''

    df = parse_contents(contents, filename)
    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            style_table={'maxHeight': '300px', 'overflowY': 'scroll'},
        )
    ])


@app.callback(
    Output('optimization-results-df', 'data'),
    [
        Input('run-optimization', 'n_clicks'),
    ],
    [
        State('max-total-costs', 'value'),
        State('max-customers', 'value'),
        State('weight-revenue', 'value'),
        State('weight-retention', 'value'),
        State('upload-data', 'contents'),
    ]
)
def run_optimization(n_clicks, max_total_costs, max_customers, weight_revenue, weight_retention, contents):
    if n_clicks is None or contents is None:
        return None

    df = parse_contents(contents, 'uploaded_file')

    result = maximize_marketing_value_multi_objective(df, max_total_costs, max_customers, weight_revenue, weight_retention)

    # Convert DataFrame and result to native Python types
    df = df.applymap(lambda x: x.item() if isinstance(x, np.int64) else x)
    result = {key: value.item() if isinstance(value, np.int64) else value for key, value in result.items()}

    # Create a list of dictionaries for each selected customer
    selected_customers_data = [
        {
            'ObjectiveRevenue': result['ObjectiveRevenue'],
            'ObjectiveRetention': result['ObjectiveRetention'],
            'SelectedCustomer': customer,
            'ChannelCost': result['ChannelCosts'][customer],
            'EngagementFraction': result['EngagementFractions'][customer],
        }
        for customer in result['SelectedCustomers']
    ]
    
    return selected_customers_data




if __name__ == '__main__':
    app.run_server(debug=True)
