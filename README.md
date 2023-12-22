# Movies in time

***Abstract***

We think that there are some correlations between a movie’s success (box office, Oscars and rating) and its release date within different time-frames and that the producers know it, and exploit it. There are some obvious trends that are common knowledge such as the blockbuster season, but we would like to look at more specific, local and unexplained trends in movie releases. Our goal is to show how we are influenced by time in our perception of movies, and what are the strategies movie studios use to benefit from this bias or what they get wrong, whether it's expressed within a month, a year, a decade or a lifetime. This could then be useful data for studios on what to avoid and what to look for when choosing the release date of a movie. To this end, we need to ask ourselves a couple of questions.


## Research questions

- Can we find any trends in movie release dates within a month, a year, and are there dates or periods that producers avoid ?

- What are the correlations between the release month and the box office performance ? Do these correlations change across history ? 

- Is there a causal effect of the release month on the box office performance ? If yes, does it change across countries ? 

- Is there a causal effect of the period of release (week, month, day of the week, etc.) on the box office performance ? 

- Is there a causal effect of the season of release on the amount of Oscars won ? What is the best way to define season in that context (i.e. what are the best dates that constitute a season when it comes to Oscar winning trends) ?


## Proposed additional datasets
- "The Movies Dataset" from Kaggle : This dataset’s content is very similar to the CMU dataset’s, which here works to our advantage because we used to complete missing data in our dataset. This allows us to work with a more robust dataset and broaden the scope of our analysis.

- "IMDB Dataset" from Kaggle : Gross isn’t a sufficient metric to evaluate public perception, which in our analysis, is critical. That’s where this additional dataset comes in. It offers key indicators of a movie’s success not present in our initial dataset, such as average rating and the volume of votes a movie received.

- "The Oscar Award, 1927-2023" from Kaggle : This dataset features Oscar wins and nominations, which is a significant measure of a movie's acclaim and success. Moreover, this attribute is often closely correlated with the film's release date, offering a temporal dimension to the success metrics in our analysis. This dataset contains the movies rewarded each year and for each category.

## Methods

Preprocessing: One obviously important component of our analysis is dates, so a big part of our preprocessing phase was converting our dates into other formats. We added to our original dataframe columns giving us information the release date’s day of the week, position in the year and week number. Having access to these new formats will facilitate further analysis.


### OLS
To do Best Release date analysis, we use OLS to research the correlation between the release date and the box office. Ordinary least square is a way to find the relationship between the dependent variables (i.e. Box office) and independent variables (i.e. dummies variables of the month). Here, we use December as a benchmark; in other words, only a constant term and dummy variable for January to November will be included in the independent variables. We run the regression for the whole data, on different countries and different time intervals. We will monitor the t-values of the coefficients and R-square of each regression. And we will try to add some independent variables to eliminate the effect of "confounder".

### Matching OLS
To do deeper research on Box office Causal analysis, we use "matching OLS". We will make an example to clarify the method we use. Suppose we research the data for the US in January. Firstly, we will get the data only for the US and split it into two groups. One is in January (control group), and the treatment group is for all other months except January. Then, we match the control and treatment groups on all relevant covariates to have them be the same size. Finally, we will do the regression below. Note that most countries will not have enough movies in our dataset to conduct this matching and get some statistically significant results. Despite this, those countries are still interesting to visualize.


### PCA
Principal components analysis is a dimensionality reduction and signal enhancement technique that will let us look at the most common variations within our dataset. Obvious trends in movie releases across the year are easy to observe and analyze, for example the summer blockbuster season. But smaller, localized and significant variations can be easy to miss. Here we’ll use PCA on the release dates and ratings of each year to spot these trends, how they change across genre, time and place, and how they can be explained.

## Proposed timeline

week 0. 13 Nov 2023 ~ 19 Nov 2023\
week 1. finish homework 2 before Friday. Have a meeting with the supervisor on Friday to talk about the research method\
week 2. learn about how to create a website using github and go deeper into our analysis. \
week 3. get feedback from milestone 2 and precise our research questions. \
week 4. devolpe the website and finalize the analysis. \
week 5. cleanup of the jupyter notebook, finish website and readme.

Note: have meeting with the superviser each week.

## Organization whithin the team

Haoyu Wen : Ordinary least square \
Florian Delavy : Genres analysis \
Leo Sperber : Principal component analysis \
Robin Patriarca: Data visualization and preprocessing

## Contributions
Haoyu Wen : Best Release date analysis, Part of Box office Causal analysis \
Florian Delavy : Genres analysis and notebooks merging \
Leo Sperber : Website, data story, world map and PCA \
Robin Patriarca: Oscar Analysis + causal box office analysis (and corresponding visualizations)
