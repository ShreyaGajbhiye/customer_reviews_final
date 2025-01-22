import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import json
from openai import AzureOpenAI
from visualization import generate_wordcloud_image
from visualization import get_top_keywords
from visualization import red_color_func
from visualization import generate_keyword_bar_chart

config_path = 'config.json'
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

azure_api_key = config['azure_api_key']
azure_api_version = config['azure_api_version']
azure_endpoint = config['azure_endpoint']
deployment_name = config['deployment_name']

client = AzureOpenAI(
    api_key=azure_api_key,  
    api_version=azure_api_version,
    azure_endpoint=azure_endpoint
)

def generate_text(reviews_text):
    prompt_text = f"""
    Please analyze the following customer review(s) about a business and extract the information below. 
    Provide the results in **valid JSON format**. For each category ("WhyCustomersLove", "PainPoints", and "Improvements"), list all relevant points with detailed explanations and specific examples from the reviews. Do not add whole reviews in the text. Include as many points as are applicable based on the reviews.

    - If a category has **no relevant points**, please provide a message:
    - For **WhyCustomersLove**, use: "A chance to shine! Enhancing the customer experience could lead to more favorable feedback."
    - For **PainPoints**, use: "Fantastic news!Customers are pleased, with no notable complaints or concerns. "
    - For **Improvements**, use: "Keep up the good work! No improvements needed at this time; continue providing excellent experiences"

    **Fields to extract:**

    {{
        "Summary": {{
            "WhyCustomersLove": [
                "List of reasons why customers love the business, each with detailed explanations and specific examples."
            ],
            "PainPoints": [
                "List of customer's pain points, each with detailed explanations and specific examples."
            ],
            "Improvements": [
                "List of suggestions for what the business can improve, each with detailed explanations and specific examples."
            ]
        }}
    }}

    **Example Output:**

    {{
        "Summary": {{
            "WhyCustomersLove": [
                "Unlimited soup, salad, and breadsticks - Many reviewers mention loving the unlimited salad, breadsticks, and soup options. They appreciate the value and the ability to enjoy these popular items as much as they want. The salad is frequently described as fresh and delicious, while the breadsticks are praised for being warm, buttery, and garlicky.",
                "Friendly and attentive service - Multiple reviews highlight positive experiences with servers who are described as friendly, attentive, and accommodating. Reviewers appreciate servers who check on them regularly, keep drinks refilled, and go above and beyond to ensure a good experience. Some even mention servers by name who made their visit especially enjoyable.",
                "Large portions and good value - Several reviewers comment on the generous portion sizes, noting they often have plenty of leftovers to take home. The ability to get a filling meal at a reasonable price point is seen as a good value by many customers. The $5 take-home meal deal is also mentioned as a popular option.",
                "Tasty Italian-American comfort food - While not gourmet Italian cuisine, many reviewers enjoy the familiar pasta dishes, soups, and other Italian-American comfort foods. Particular favorites mentioned include the chicken gnocchi soup, fettuccine alfredo, lasagna, and chicken parmesan. The food is often described as reliably tasty.",
                "Family-friendly atmosphere - Several reviewers note that the restaurant provides a welcoming environment for families with children. They appreciate touches like coloring books, crayons, and electronic tablets to keep kids entertained. The casual atmosphere and kid-friendly menu options make it a go-to choice for family meals out."
            ],
            "PainPoints": [
            "Long waiting times during peak hours - Some reviewers express frustration with the lengthy wait times for a table, especially during dinner hours and weekends. They mention waits ranging from 30 minutes to over an hour, which affected their overall dining experience.",
            "Inconsistent food quality - A number of customers note that the quality of food can be inconsistent. While some dishes meet expectations, others arrive lukewarm or overcooked. For example, a reviewer mentioned their pasta was undercooked and lacked sufficient sauce.",
            "Slow service - Several reviews point out that service can be slow, particularly when the restaurant is busy. Customers report long waits for their orders to be taken, delays in receiving food, and difficulty getting the attention of servers for refills or additional requests.",
            "Noisy environment - Some patrons find the restaurant to be quite noisy, making it hard to have conversations. They cite loud background music and close seating arrangements as contributing factors to the high noise levels.",
            "Cleanliness issues - A few customers have raised concerns about cleanliness, mentioning dirty utensils or unclean tables. One reviewer noted that their table had not been wiped properly and still had crumbs from previous guests."
        ],
        "Improvements": [
            "Reduce wait times - Implementing a better reservation or seating system could help minimize long waiting periods. For instance, offering call-ahead seating or improving the efficiency of table turnover might enhance customer satisfaction.",
            "Ensure consistent food quality - Focusing on kitchen staff training and quality control can help maintain a consistent standard across all dishes. Regular checks might prevent issues like undercooked pasta or cold meals.",
            "Improve service speed - Increasing staff during peak hours or optimizing workflow processes could reduce delays in service. Ensuring that servers are attentive even when busy can enhance the overall dining experience.",
            "Manage noise levels - Adjusting the volume of background music and considering interior design changes, such as adding sound-absorbing materials, could create a more comfortable atmosphere for conversation.",
            "Enhance cleanliness standards - Implementing stricter cleaning protocols and regular inspections can address concerns about cleanliness. Training staff to promptly clean tables and check utensils before setting them can improve customer perceptions."
        ]
            
    }}
    }}
        {{
        "Summary": {{
        "WhyCustomersLove": [
            "Lightweight and portable - Customers value the product's ease of transport due to its compact and lightweight design.",
            "Powerful performance - Equipped with advanced features or technology, it handles its intended tasks efficiently.",
            "High-quality output - The product delivers excellent results, whether in performance, visuals, or functionality.",
            "Long-lasting battery life (if applicable) - Users appreciate extended usage times without the need for frequent recharging.",
            "User-friendly interface - The intuitive design and easy-to-use controls enhance the overall user experience."
        ],
        "PainPoints": [
            "Limited compatibility or connectivity - The product may lack compatibility with certain accessories or require additional adapters.",
            "Noise during operation - Some reviewers report that the product can be loud or distracting when in use.",
            "High cost - The premium features come with a steep price tag that may be a barrier for some customers.",
            "Short warranty period - Customers desire longer warranty coverage to protect their investment.",
            "Subpar components or features - Certain aspects of the product may not meet customer expectations (e.g., camera quality, screen resolution)."
        ],
        "Improvements": [
            "Enhance compatibility or connectivity options - Including more ports or compatibility features would improve usability.",
            "Optimize operational noise levels - Improving the design can reduce noise during use.",
            "Offer flexible pricing or financing - Providing payment plans or reducing the price can make the product more accessible.",
            "Extend warranty coverage - A longer warranty period would give customers greater confidence in their purchase.",
            "Upgrade key components or features - Enhancing specific parts of the product aligns with customer needs and expectations."
        ]
    }}
    }}

    **Review(s):**

    "{reviews_text}"
    """

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are an analytical assistant that provides detailed summaries of restaurant customer reviews. Extract key insights and present them in a structured JSON format, including specific examples and organizing points as lists where appropriate without directly quoting the reviews."},
            {"role": "user", "content": prompt_text}
        ],
        max_tokens=1000,
        temperature=0.2,#temperature should be low as the output should be same for the same input
        top_p=1.0, # all the words are taken into consideration, if 0.9 then 90% of the top words are taken into consideration
        frequency_penalty=0, #used to control the repetition of the words
        presence_penalty=0 
    )
    text = response.choices[0].message.content.strip()
    print("Raw API Response:", text)
    try:
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON. Response was not valid JSON.", e)
        result = {"error": "Invalid JSON", "response": text}
    except Exception as e:
        print("API call error:", e)
        result = {"error": "API call failed", "response": str(e)}

    return result

 

# Load restaurant reviews data
df = pd.read_csv("common.csv")


# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([

    html.H1("Business Review Analysis"),
    
    html.Label("What Industry Are You In?"),
    dcc.Dropdown(
        id='business-type-dropdown',
        options=[{'label': business, 'value': business} for business in df['Business_Type'].unique()],
        value=None,  # Default value
        placeholder="Select an Industry",
        clearable=True
    ),
    html.Label("Pick a Business/Product from the List:"),
    dcc.Dropdown(
        id = 'business-name-dropdown',
        placeholder = "Select a Business/Product Name",
    ),
    html.Hr(),
    dcc.Loading(
        id="loading-icon",
        type="default",
        fullscreen = True,  # You can change this to 'default' or 'dot'
        children=[
            html.Div(id='loading-message', children="Fetching insights... Please wait while we analyze the reviews."),
            html.Div(id='review-summary')
        ]
    )
])

@app.callback(
        Output('business-name-dropdown','options'),
        Input('business-type-dropdown','value')
)
def update_business_name_dropdown(selected_business_type):
    if selected_business_type:
        filtered_df = df[df['Business_Type'] == selected_business_type]
        return [{'label' : name, 'value': name} for name in filtered_df['Business_Name'].unique()]
    return []
    
def format_point(point):
        if isinstance(point, dict):
            subheading = point.get('suggestion', point.get('issue', 'Unknown Issue'))
            description = point.get('details','No further details provided.')
            return html.Li([html.Strong(subheading + " - "), description])
        elif isinstance(point, str):
            if ' - ' in point:
                subheading, description = point.split(' - ', 1)
                return html.Li([html.Strong(subheading + " - "), description])
            else:
                return html.Li(point)
        else:
            return html.Li("Invalid data format")



@app.callback(
    [Output('review-summary', 'children'),
     Output('loading-message', 'children')],
    [Input('business-name-dropdown', 'value')]
)

def update_review_summary(selected_business_name):
    if not selected_business_name:
        return html.P("Please select a business to see the review summary."), "Waiting for selection..."

    reviews = df[df['Business_Name'] == selected_business_name]['Review'].tolist()
  
    reviews_text = " ".join(reviews)
    
    
    summary = generate_text(reviews_text)
    
  

    print("Summary Response:", summary)

    if 'error' in summary:
        return html.P(f"Error: {summary['error']}"), "Failed to fetch insights. Please try again."
    

    why_customers_love = summary.get("Summary", {}).get("WhyCustomersLove", [])
    pain_points = summary.get("Summary", {}).get("PainPoints", [])
    improvements = summary.get("Summary", {}).get("Improvements", [])
    
    # Generate word clouds for each category
    why_customers_love_wc = generate_wordcloud_image(why_customers_love, ngram_range=(1, 2))
    pain_points_wc = generate_wordcloud_image(pain_points, ngram_range=(1, 2),color_func='red')
    improvements_wc = generate_wordcloud_image(improvements, ngram_range=(1, 2))

   # Generate top keywords and bar chart image
    top_keywords = get_top_keywords(reviews, top_n=20, ngram_range=(2,2))
    keyword_bar_chart_img = generate_keyword_bar_chart(top_keywords)

    def format_point(point):
        if isinstance(point, dict):
            subheading = point.get('suggestion', point.get('issue', 'Unknown Issue'))
            description = point.get('details','No further details provided.')
            return html.Li([html.Strong(subheading + " - "), description])
        elif isinstance(point, str):
            if ' - ' in point:
                subheading, description = point.split(' - ', 1)
                return html.Li([html.Strong(subheading + " - "), description])
            else:
                return html.Li(point)
        else:
            return html.Li("Invalid data format")
        


    def list_or_text(section):
        if isinstance(section, list):
            return html.Ul([format_point(point) for point in section])
        elif isinstance(section, str):
            return html.P(section)

    # Return the summary and word clouds
    return (
        html.Div([
            html.H2("Summary of Customer Reviews"),
            html.Div([
                html.H3("Why Customers Love Us"),
                list_or_text(why_customers_love),
                
                html.H3("Customer Pain Points"),
                list_or_text(pain_points),
                
                html.H3("Suggested Improvements"),
                list_or_text(improvements),
            ], style={'margin-bottom':'50px'}),

            # Embed the keyword bar chart image
            # html.H2("Top Keywords in Reviews"),
            html.Div([
                html.Img(src=keyword_bar_chart_img, style={'width': '60%', 'max-width': '600px',  'height': 'auto','margin-left': 'auto', 'margin-right': 'auto', 'border': '1px solid #ccc', 'display': 'block'})
            ], style={'text-align': 'center', 'padding': '10px'}),
        

            # html.H2("Take a Look at These Informative Images:"),
            html.Div([
                html.Div([
                    html.H4("Why Customers Love You"),
                    html.Img(src=why_customers_love_wc, style={'width': '100%', 'border': '1px solid #ccc'})
                ], style={'flex': '1', 'padding': '10px'}),

                html.Div([
                    html.H4("Customers' Pain Points"),
                    html.Img(src=pain_points_wc, style={'width': '100%', 'border': '1px solid #ccc'})
                ], style={'flex': '1', 'padding': '10px'}),

                html.Div([
                    html.H4("Suggested Improvements"),
                    html.Img(src=improvements_wc, style={'width': '100%', 'border': '1px solid #ccc'})
                ], style={'flex': '1', 'padding': '10px'}),
            ], style={
                'display':'flex', 
                'justify-content':'space-between',
                'align-items':'center'
            })
        ]),
        "Insights fetched successfully."
    )
    
    
    # # Create HTML components for better formatting
    # return html.Div([
    #     html.H3("Why Customers Love Your Business"),
    #     html.Ul([format_point(point) for point in why_customers_love]) if why_customers_love else html.P("A chance to shine! Enhancing the customer experience could lead to more favorable feedback."),

    #     html.H3("Customer's Pain Points"),
    #     html.Ul([format_point(point) for point in pain_points]) if pain_points else html.P("Fantastic news! Customers are pleased, with no notable complaints or concerns."),

    #     html.H3("Suggessted Improvements"),
    #     html.Ul([format_point(point) for point in improvements]) if improvements else html.P("Keep up the good work! No improvements needed at this time; continue providing excellent experiences.")
    # ]),"Insights fetched successfully."
    # Create HTML components for better formatting
    # formatted_output = html.Div([
    #     html.H3("Why Customers Love your Business"),
    #     html.Ul([format_point(point) for point in why_customers_love]),

    #     html.H3("Pain Points of Customers"),
    #     html.Ul([format_point(point) for point in pain_points]),

    #     html.H3("What can be Improved"),
    #     html.Ul([format_point(point) for point in improvements])
    # ])

    # return formatted_output, ""


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
