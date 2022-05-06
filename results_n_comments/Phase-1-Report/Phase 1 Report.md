# Phase 1 Report - Power Consumption in Remote Northern Communities

>Alireza Yazdi, Ph.D. & LingJun Zhou, Ph.D. @Digital Accelerator

> Date: 

# Outline
- Introduction & Background
- Past work
- Approaches
- Results
- Discussion & Prospects

## Introduction & Background

Natural Resources Canada has been dedicated to facilitate the transition from traditional fossil fuel based power to renewable, green powers in Canadian northern communities. Most of these communities are geologically isolated and the only access to electric power comes from fossil fuel generators. In this initiative, we aim to accurately forecast the annual power consumption given a historical power consumption data with hourly resolution. The input and output of this project are both power consumptions. 
## Past work
- Alexei,
- Jaret Andre,
- Krishan Rajaratnam, 


## Approaches

Following CanmetENERGY's initial trial model developed by Ryan Kilpatrick, the authors were given three years (2013, 2014, 2015) of energy usage data and were tasked to forecast the 4th year (2016). 

- Initial approach:
    - Given the fact that there are three years worth of data and each year we have the same amount of hours, Ryan proposed to line up each year's data side by side by the hour of the year, then he proceeded to take the average of each hour's power usage and normalized/standardized the average power consumption of that hour with peak power usage. He then utilized the total fuel consumption to rescale the standardized hourly power usage back to an hourly power consumption (of that hour), which serves as the predicted hourly power consumtion. 

- Alireza's approach:
    - (Please fill in)

- LingJun's approach:
    - Given the hourly power consumption data, LingJun first remvoed the population factor by calculating the power consumption per capita. He then plotted them over time and observed an annual seasonality, namely, the annual ups and downs of the power consumption seems to repeat itself. The power consumption data evidently has three components: trends, seasonality, and noise. He then used a sliding window approach to tabularize the power consumption data and made use simple linear regression as forecaster to forecast an entire year worth of power consumption to hourly precision. 
    - A good example of how this process works can be described as below:
        1. First, the sliding window takes the past, say 24 hours of power consumption and a linear regressor to predict the 25th hour of power consumption.
        2. Second, the sliding window moves forward by 1 step to include the predicted 25th hour's powe consumption, but gets rid of the 1st hour's data (so that the window stays size 24) and predicts the 26th hour's power consumption.
        3. This process continues till we have an entire year's power consumption forecast.

# Results

- Ryan's approach:
    - Ryan's approach so far has the best "prediction" in terms of errors (RMSE). This could be resulting from the simplicity of his approach, but also could be a result of potential data leakage. Considering that Ryan's used all three years' data as input and try to "predict" a typical year's power consumption. The input and output might have some overlap.

- Alireza's approach:
    - (Please fill in)

- Lingjun's approach:
    - LingJun's approach is very quick to train and yield fairly good results. The actual results can be shown in the image below. 
    
    ![A typical year's forecast](A_typical_year_s_forecast_LingJun.PNG)
    The x-axis is hour counts, the y-axis is power consumption per capita, and `MAE` is Mean Absolte Error; `MAPE` is Mean Absolute Percentage Error; `RMSE` is Root Mean Squared Error.

## Discussion & Prospects

- Discussion:
    - The authors have observed that although a typical year's power consumption data has very distinguishable seasonality, some communities' data do pose a threat to this. For example, for `Ivujivik`, the second year data is quite a-typical, in the sense that there are many ups and downs that defy seasonality. Please refer to the image below. 
    ![An atypical year's forecast](An_atypical_year_s_forecast_LingJun.PNG)
    We clear see the 2nd year does not have any visual pattern, which potentially mean this entire year at `Ivujivik` could be outliers. 

    - #### Results comparison `2015 RMSE comparison`

        |Community	|RMSE-Ryan	| RMSE-Ali	|RMSE-LingJun|
        |---------|-------|--------|------------|
        |Inukjuak	|73.5	|97.5	|99.237
        Salluit	|57.9|	78.2|	82.08027229
        Quaqtag	| 25.9	|35.1	|37.10398047
        Aupaluk	|21.8	|22.4	|23.6506428
        Ivujivik|	23.3|	45.1|	49.84214596
        Kuujjuaq|	172	|143.3	|165.2887934
        Kangirsuk|	40.6|	49.8|	61.79603198
        Kuujjuarapik	|123.7	|87.2	|91.28434919
        Kangiqsualujjuaq	|58.7|	54.8	|56.739242

- Prospects:
    - 
