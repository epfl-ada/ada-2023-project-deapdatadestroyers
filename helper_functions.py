import pandas as pd
import numpy as np
import scipy
import seaborn as sbn
import matplotlib.pyplot as plt
import json

def count_var(df, var, show_graph=True, show_chisquare=True):
    """Plots the number of row (movies) for each category of a variable (Month, Weekday Name, Day, Year)

    Args:
        df (data frame): data frame
        var (string): variable to plot
        show_graph (bool, optional): show the graph. Defaults to True.
        show_chisquare (bool, optional): show the chisquare statistic. Defaults to True.
    
    Returns:
        var_values (array): number of movies for each category of the variable
    """
    
    df_temp = df.copy(deep=True)
    df_temp.dropna(subset=[var],inplace=True)

    var_values = df_temp.groupby(var).count()['Name'].values
    
  
    if show_chisquare:
        print(f'Chisquare statistic : {scipy.stats.chisquare(var_values)}')
    if show_graph:
        order=None
        if var == 'Month':
            order = np.linspace(1,12,12).astype(int)
        if var == 'Weekday Name':
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if var == 'Day':
            order = np.linspace(1,31,31).astype(int)
        sbn.set(rc={'figure.figsize':(10,4)})
        
        ax = sbn.countplot(x=var, data=df_temp, order=order)
        
        if var == 'Year':
            plt.xticks(rotation=45, ha='right')
            # Keep only the labels for every five year 
            labels = [item.get_text() for item in ax.get_xticklabels()]
            ax.set_xticklabels([label if int(label) % 5 == 0 else '' for label in labels])

        plt.title(f'Number of movie premieres for category : {var}')
        plt.xlabel(f'{var}')
        plt.ylabel('Number of movies')
        plt.show()
    
    return var_values

def count_var_normalized_genre(initial_df, genre_df, var, show_graph=True, show_chisquare=True):
    """Plots the percentage of row (movies) for each category of a variable (Month, Weekday Name, Day, Year) for a specific genre
    """

    df_genre_temp = genre_df.copy(deep=True)
    df_total_temp = initial_df.copy(deep=True)
    
    df_genre_temp.dropna(subset=[var],inplace=True)
    df_total_temp.dropna(subset=[var],inplace=True)

    genre_values = df_genre_temp.groupby(var).count()['Name']
    total_values = df_total_temp.groupby(var).count()['Name']
    
    proba = ((genre_values/ total_values) * 100).fillna(value= 0)
    proba_values = proba.values
    
  
    if show_chisquare:
        print(f'Chisquare statistic : {scipy.stats.chisquare(proba_values)}')
    if show_graph:
        order=None
        if var == 'Month':
            order = np.linspace(1,12,12).astype(int)
        if var == 'Weekday Name':
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if var == 'Day':
            order = np.linspace(1,31,31).astype(int)
        sbn.set(rc={'figure.figsize':(10,4)})
        
        ax = sbn.barplot(x=proba.index, y=proba.values)
        
        if var == 'Year':
            plt.xticks(rotation=45, ha='right')
            # Keep only the labels for every five year 
            labels = [item.get_text() for item in ax.get_xticklabels()]
            ax.set_xticklabels([label if int(label) % 5 == 0 else '' for label in labels])

        plt.title(f'Percentage of movie premieres for category : {var}')
        plt.xlabel(f'{var}')
        plt.ylabel('Percentage of movies')
        plt.show()
    
    return proba_values


def avg_var(df, var='Box office', group='Month', show_graph=True, logscale=True):
    """
    Plots the average of a variable (Box office, Metascore, IMDB score) for each category of a variable (Month, Weekday Name, Day, Year)
    
    Args:
        df (data frame): data frame
        var (string): variable to plot
        group (string): variable to group by
        show_graph (bool, optional): show the graph. Defaults to True.
        logscale (bool, optional): use logscale for the y-axis. Defaults to True.
    """
    
    df_temp = df.copy(deep=True)
    df_temp.dropna(subset=[group, var],inplace=True)
    
    df_mean = df_temp.groupby(group)[var].mean().values
    df_var = list(df_temp.groupby(group).groups.keys())

    # Graph
    
    if show_graph:
        sbn.set(rc={'figure.figsize':(10,4)})
        order=None
        if group == 'Month':
            order = np.linspace(1,12,12).astype(int)
        if group == 'Weekday Name':
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if group == 'Day':
            order = np.linspace(1,31,31).astype(int)
        # No boxplot for year, too many boxes makes it unreadable   
        
        sbn.boxplot(x=group, y=var, order = order, data=df)
        plt.title(f'{var} for movies by {group}')
        plt.xlabel(f'{group}')
        plt.ylabel(f'Average {var}')
        if logscale:
            plt.yscale("log")
        plt.show()
    
    return df_mean

def get_movies_genre(df, genre):
    df_temp = df.copy(deep=True)
    df_temp = df_temp[df_temp['Genres'].apply(lambda x: genre in x)] 
    return df_temp

def get_movies_country(df, country, contains=True):
    df_temp = df.copy(deep=True)
    df_temp = df_temp[(df_temp['Countries'].apply(lambda x: country in x))==contains]
    return df_temp


def new_date_format(df_movie):
    """
    This function is to convert the year, month and day into date format and week number.
    """

    df_movie_PCA = df_movie.dropna(subset=['Year', 'Month', 'Day'])
    df_movie_PCA = df_movie_PCA.reset_index(drop=True)


    df_movie_PCA['Release_Date'] = pd.to_datetime(df_movie_PCA[['Year', 'Month', 'Day']], errors='coerce')


    df_movie_PCA['Week_Number'] = df_movie_PCA['Release_Date'].dt.strftime('%V')

    
    df_movie_PCA['Week_Number'] = pd.to_numeric(df_movie_PCA['Week_Number'], errors='coerce').astype('Int64')

    return df_movie_PCA



def normalize_matrix(main_variations, new_min=-1, new_max=1):
   
    row_mins = np.min(main_variations, axis=1, keepdims=True)
    row_maxs = np.max(main_variations, axis=1, keepdims=True)

    normalized_matrix = ((main_variations - row_mins) / (row_maxs - row_mins)) * (new_max - new_min) + new_min

    return normalized_matrix



def get_values_column_of_list(df, var='Genres'):
    values = dict()
    for i, element in enumerate(df[var]):
        for v in element:
            if v not in values:
                values[v] = 0
            values[v] = values[v] + 1
    return values



def get_list_of_countries(df_series):
    """This is the function to get the list of countries' name from a dataframe series.

    Args:
        df_series (data frame series): the column of data frame series include the countries name

    Returns:
        list_of_countries (list): list of countries name
    """
    #The following step is to find the list of all the countries name which is show in the 'Countries (Freebase ID:name tuples)' column
    # Define a function to extract country names
    def extract_countries(location):
        # Convert the string to a dictionary
        location_dict = json.loads(location.replace("'", "\""))
        # Extract and return the country names
        return list(location_dict.values())
    #transfer the text of 'Countries (Freebase ID:name tuples)' into some string only contain the countries name
    df_aux = df_series.apply(extract_countries)
    df_aux = pd.DataFrame(df_aux.apply(lambda x: ', '.join(x) if x else None))
    df_aux = df_aux.dropna()
    #list the unique countries name.
    list_of_countries = df_aux['Countries (Freebase ID:name tuples)'].unique()
    string_of_countries = ','.join(list_of_countries)
    list_of_countries = string_of_countries.split(',')
    list_of_countries = [x.strip() for x in list_of_countries]
    list_of_countries = list(set(list_of_countries))
    return list_of_countries


def box_month_plot(height, width, dictionary, months, f_h, f_w):
    """Plot the coefficient of regression

    Args:
        height (int): number of subplot in vertical axis
        width (int): number of subplot in horizontal axis
        dictionary (dict): dictionary to store the coefficient and t-values
        months (list): list of month
        f_h (int): height of figure size
        f_w (int): wight of figure size 
    
    """
    # Create a subplot with 4 rows and 2 columns
    fig, axes = plt.subplots(height, width, figsize=(f_h, f_w))
    axes = axes.flatten()

    # Create bar charts for each country
    for i, name in zip(np.arange(0,height*width),dictionary.keys()):
        ax = axes[i]
        ax.bar(months, dictionary[name][0])
        if isinstance(name,int) or isinstance(name,np.int64) :
            ax.set_title('coefficient of regression Monthly \n dummies variable to Box office w.r.t {}-{}'.format(name, name+20))
        else:
            ax.set_title('coefficient of regression Monthly \n dummies variable to Box office w.r.t {}'.format(name))
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        ax.set_ylabel('coefficient')
        ax2 = ax.twinx()
        
        # Plot the t-values and 1.96 and -1.96 to check whether the coefficient is significant.
        ax2.plot(months, dictionary[name][1], color='orange', marker='o', label='t-values')
        ax2.plot(months, 12*[1.96], color='red', label='1.96')
        ax2.plot(months, 12*[-1.96], color='red', label='-1.96')
        ax2.set_ylabel('T-values')
        ax2.legend(loc='upper right')
        
        # Align the zero point for the left y-axis and right y-axis.
        aL, aaL = ax.get_ylim()
        aR, aaR = ax2.get_ylim()
        Left = ax.get_yticks()
        Right = ax2.get_yticks()
        aaL = max(aaL,abs(aL))
        aL = min(-aaL,aL)
        aaR = max(aaR,abs(aaR))
        aR = min(-aaR,aR)
        ax.set_ylim(aL,aaL)
        ax2.set_ylim(aR,aaR)
        ax.set_yticks(Left)
        ax2.set_yticks(Right)

    plt.tight_layout()
    plt.show()


def get_time_stamps_df(df):
    df_time_stamps = df.copy(deep=True)

    df_time_stamps.dropna(subset=['Month'], inplace=True)
    temp = (df_time_stamps['Year'].astype(str) + '-' 
                                                 + df_time_stamps['Month'].astype(str) + '-' 
                                                 + df_time_stamps['Day'].astype(str))


    df_time_stamps['Weekday'] = temp.apply(lambda x: pd.to_datetime(x, errors = 'coerce').dayofweek)
    df_time_stamps.dropna(subset=['Weekday'], inplace=True)
    df_time_stamps['Weekday'] = df_time_stamps['Weekday'].apply(int)


    df_time_stamps['Weekday Name'] = temp.apply(lambda x: pd.to_datetime(x, errors = 'coerce')).dt.day_name()

    
    return df_time_stamps


def extract_country_name(country_string):
    try:
        # Find the position of ": " and the position of the second key
        middle_index = country_string.find(',')
        start_index = country_string.find(': "') + 3
        end_index = country_string.find('"', start_index, middle_index)

        # Extract the substring between ": " and the second key
        country_name = country_string[start_index:end_index]

        return country_name
    except Exception as e:
        return "Unknown"
    
    
COUNTRY_CODE = {
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
    "Korea, South": "KOR",
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

def control_repartition(control_var, df_treatment, df_control):
    for var in control_var:
        df_treatment_var = df_treatment.copy(deep=True).dropna(subset=var)
        df_control_var = df_control.copy(deep=True).dropna(subset=var)
        e1 = df_control_var[var].mean()
        e2 = df_treatment_var[var].mean()
        var1 = df_control_var[var].var()
        var2 = df_treatment_var[var].var()
        
        SMD = abs(e1-e2)/(np.sqrt(var1 + var2))

        if SMD >= 0.1:
            print(f"Careful, not a good balance for {var} because SMD = {SMD} (>0.1)")
        else:
            print(f"Good balance for {var} because SMD = {SMD} (<0.1)")
            
def add_dummies(df, var, top_dummies):
    df_dummies = df.copy(deep=True)

    for value in top_dummies:
        df_dummies[str(value).replace(' ', '_').replace('-','_') + '_onehot'] = df_dummies[var].apply(lambda x: 1 if value in x else 0)
    #df_dummies.drop(labels=var, axis=1, inplace=True)
    return df_dummies