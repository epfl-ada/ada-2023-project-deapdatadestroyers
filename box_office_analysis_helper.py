from helper_functions import *
from matching import *
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import math



def create_data_for_graph_with_dropdown(df_time_stamps, scatter_dot_table, scatter_dot_table_name, timetable, 
                                        timetable_name, dependent_var, matching_vars,  onehot_vars):

    # Exploding for genres and making linear regression for each genre
    for i, item in enumerate(scatter_dot_table):
        for j, timeframe in enumerate(timetable):

            if scatter_dot_table_name == 'Country':
                df_country = get_movies_country(df_time_stamps, item)
            elif scatter_dot_table_name == 'Genre':
                df_country = get_movies_genre(df_time_stamps, item)
    
            df_country['is_timeframe'] = (df_country[timetable_name] == timeframe).astype(int)
            df_country_time_frame = df_country[df_country['is_timeframe']==1]
            print(timeframe, j)
            #display(df_country)
    
            print(f"For {item}")
        
            (df_matched_treatement, df_matched_control, 
             p_values, effect_sizes, intercept) = analyse(df_country, 'is_timeframe', matching_vars, dependent_var, 'Wikipedia ID', onehot_vars)
            if effect_sizes == np.NaN:
                abs_effect_size = np.NaN
            else:
                abs_effect_size = abs(effect_sizes)
            
            if (i, j) == (0, 0):
                data = []
            mean_value = df_country_time_frame[dependent_var].mean()
            if pd.isnull(mean_value):
                mean_value = 0
            data.append([item, timeframe,mean_value , df_country['is_timeframe'].sum(), 
                         p_values, effect_sizes, abs_effect_size, intercept])
            
    
    df_cluster_graph = pd.DataFrame(data, columns = [scatter_dot_table_name, timetable_name, 'Average ' + dependent_var, 'Number of movies', 'p_value', 
                                                     'effect_size', 'abs_effect_size', 'Intercept'])
    #display(df_cluster_graph)

    return df_cluster_graph






def create_figure_no_dropdown(df_cluster_graph, hover_data, bubble_size, bubble_color, x, y, x_log=True, title = "yoyoyo"):
    fig = px.scatter(df_cluster_graph, x=x, y=y, #color=bubble_color, 
                         custom_data=hover_data, size=bubble_size)


    fig.update_layout(width=800, height=500, title_font=dict(size=20), title = title,
                      font=dict(family='Arial', size=12),
                      xaxis=dict(title=x, titlefont=dict(size=14)),
                      yaxis=dict(title=y, titlefont=dict(size=14)),
                      showlegend=True)
    for trace in fig.data:
            template = ''
            for i, data in enumerate(hover_data):
                if str(data)=='Country':
                    template = template + '<br><b>' + str(data) + '</b>: %{customdata['+str(i)+']}</br>'
                else:
                    template = template + '<br><b>' + str(data) + '</b>: %{customdata['+str(i)+']:,.2f}</br>'
                
            trace.hovertemplate = (template + '<extra></extra>')
 
    return fig





def create_figure_with_dropdown(df_cluster_graph, drop_down_table, drop_down_name, hover_data, bubble_size, bubble_color, x, y, x_log=True, y_log=True, title='TMP title'):

    if x_log:
        df_cluster_graph['log ' + x] =df_cluster_graph[x].apply(lambda x: np.log(x))
        x = 'log ' + x

    if y_log:
        df_cluster_graph['log ' + y] =df_cluster_graph[y].apply(lambda x: np.log(x))
        y = 'log ' + y

    if len(drop_down_table) == 0:
        fig = create_figure_no_dropdown(df_cluster_graph, hover_data, bubble_size, bubble_color, x, y, x_log, title)
        return fig
    #create a scatter plot for each genre
    figs = []
    for var in drop_down_table:
        df_subset = df_cluster_graph[df_cluster_graph[drop_down_name] == var]
        fig = px.scatter(df_subset, x=x, y=y, color=bubble_color, 
                         custom_data=hover_data, size=bubble_size)
        for trace in fig.data:
            template = ''
            for i, data in enumerate(hover_data):
                if str(data)=='Country' or str(data) == 'Genre' or str(data) == 'Weekday Name':
                    template = template + '<br><b>' + str(data) + '</b>: %{customdata['+str(i)+']}</br>'
                else:
                    template = template + '<br><b>' + str(data) + '</b>: %{customdata['+str(i)+']:,.2f}</br>'
                
            trace.hovertemplate = (template + '<extra></extra>')
        figs.append(fig)
    
        
    #combine all the figures into one
    fig = go.Figure(data=[trace for f in figs for trace in f.data])
    for i in range(len(fig.data)):
        fig.data[i].visible = (i < len(figs[0].data))


    #create the dropdown menu
    buttons = []
    for i, item in enumerate(drop_down_table):
            visibility = [False]*len(fig.data)
            visibility[i*len(figs[0].data):(i+1)*len(figs[0].data)] = [True]*len(figs[0].data)
            buttons.append(dict(label=str(item), method='update', args=[{'visible': visibility}]))
        
    fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction='down',
                    showactive=True,
                    x=0.795,
                    xanchor='left',
                    y=1,
                    yanchor='top'
                ),
            ],
        )

    
    fig.update_layout(width=800, height=500, title_font=dict(size=20), title = title,
                      font=dict(family='Arial', size=12),
                      xaxis=dict(title=x, titlefont=dict(size=14)),
                      yaxis=dict(title=y, titlefont=dict(size=14)),
                      showlegend=True)
    
    
    #fig.show(config={'displayModeBar': False})

    return fig

    #fig.write_html("../project-website/_includes/kmeans_per_genre.html", config={'displayModeBar': False})

def fig_with_flags(fig, df, size, x, y, x_log = True, y_log=True):

    fig.update_traces(marker_color="rgba(0,0,0,0)")
                      
    if x_log:
        df[x] = df[x].apply(lambda x: np.log(x))
    if y_log:
        df[y] = df[y].apply(lambda x: np.log(x))
        
    
    for i, row in df.iterrows():
        country = row['Country']
        code = COUNTRY_CODE[country][:2]
        if math.isnan(row['p_value']):
            p_value = 1
        else:
            p_value = row['p_value']
        
        #get the image dimensions
        image_size = np.sqrt(row[size] / df[size].max())*20
        
        if y_log or x_log:
            image_size = np.log(image_size)

       
        #add the main image
        fig.add_layout_image(
            dict(
                source=Image.open(f"round_flags/flags/64/{code}.png"),
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=row[x],
                y=row[y],
                opacity= (1-p_value) if p_value < 0.05 else min(1-p_value + 0.2, 0.5),
                layer="above",
                sizex=image_size,
                sizey=image_size,
                
            )
        )

    fig.update_layout({'plot_bgcolor': 'rgba(255, 255, 255, 1)'})

    fig.show(config={'displayModeBar': False})

    return fig




COUNTRY_CODE = {
    "Hong Kong": "HK",
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Andorra": "AND",
    "Angola": "AGO",
    "Antigua and Barbuda": "ATG",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "Australia": "AUS",
    "Austria": "AUT",
    "Azerbaijan": "AZE",
    "Bahamas": "BHS",
    "Bahrain": "BHR",
    "Bangladesh": "BGD",
    "Barbados": "BRB",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Belize": "BLZ",
    "Benin": "BEN",
    "Bhutan": "BTN",
    "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA",
    "Brazil": "BRA",
    "Brunei": "BRN",
    "Bulgaria": "BGR",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cabo Verde": "CPV",
    "Cambodia": "KHM",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Comoros": "COM",
    "Congo": "COG",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cuba": "CUB",
    "Cyprus": "CYP",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Djibouti": "DJI",
    "Dominica": "DMA",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI",
    "Estonia": "EST",
    "Eswatini": "SWZ",
    "Ethiopia": "ETH",
    "Fiji": "FJI",
    "Finland": "FIN",
    "France": "FRA",
    "Gabon": "GAB",
    "Gambia": "GMB",
    "Georgia": "GEO",
    "Germany": "DEU",
    "West Germany": "DEU",
    "Ghana": "GHA",
    "Greece": "GRC",
    "Grenada": "GRD",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guinea-Bissau": "GNB",
    "Guyana": "GUY",
    "Haiti": "HTI",
    "Honduras": "HND",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ireland": "IRL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kiribati": "KIR",
    "Korea, North": "PRK",
    "South Korea": "KR",
    "Kosovo": "XKX",
    "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ",
    "Laos": "LAO",
    "Latvia": "LVA",
    "Lebanon": "LBN",
    "Lesotho": "LSO",
    "Liberia": "LBR",
    "Libya": "LBY",
    "Liechtenstein": "LIE",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Maldives": "MDV",
    "Mali": "MLI",
    "Malta": "MLT",
    "Marshall Islands": "MHL",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Mexico": "MEX",
    "Micronesia": "FSM",
    "Moldova": "MDA",
    "Monaco": "MCO",
    "Mongolia": "MNG",
    "Montenegro": "MNE",
    "Morocco": "MAR",
    "Mozambique": "MOZ",
    "Myanmar": "MMR",
    "Namibia": "NAM",
    "Nauru": "NRU",
    "Nepal": "NPL",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "North Macedonia": "MKD",
    "Norway": "NOR",
    "Oman": "OMN",
    "Pakistan": "PAK",
    "Palau": "PLW",
    "Panama": "PAN",
    "Papua New Guinea": "PNG",
    "Paraguay": "PRY",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Qatar": "QAT",
    "Romania": "ROU",
    "Russia": "RUS",
    "Rwanda": "RWA",
    "Saint Kitts and Nevis": "KNA",
    "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT",
    "Samoa": "WSM",
    "San Marino": "SMR",
    "Sao Tome and Principe": "STP",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Seychelles": "SYC",
    "Sierra Leone": "SLE",
    "Singapore": "SGP",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Solomon Islands": "SLB",
    "Somalia": "SOM",
    "South Africa": "ZAF",
    "South Sudan": "SSD",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sudan": "SDN",
    "Suriname": "SUR",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tajikistan": "TJK",
    "Tanzania": "TZA",
    "Thailand": "THA",
    "Timor-Leste": "TLS",
    "Togo": "TGO",
    "Tonga": "TON",
    "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Turkmenistan": "TKM",
    "Tuvalu": "TUV",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "United Kingdom": "GBR",
    "United States of America": "USA",
    "Uruguay": "URY",
    "Uzbekistan": "UZB",
    "Vanuatu": "VUT",
    "Vatican City": "VAT",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Yemen": "YEM",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE"
}

