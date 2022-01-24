#Analysis done over baltimore's crime

#####################################################
#Libary's 
#####################################################

import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt
from sklearn.preprocessing import LabelEncoder
from itertools import product
import datetime
from datetime import datetime
from datetime import date
import math
from sklearn.linear_model import LinearRegression
import haversine as hs
import folium
import webbrowser

from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMap

#####################################################
#Loading data and some pre-work done over it
#####################################################

#--------------------------------
#crime data
#--------------------------------

#data set consisting of crimes in Baltimore from Jan 2013 to Nov 2016 - https://data.world/data-society/city-of-baltimore-crime-data
crimes_2016Novprior = pd.read_csv ('BPD_Part_1_Victim_Based_Crime_Data.csv') 

#data set consisting of crimes in Baltimore from Jan 2017 to Jan 2022 - https://data.baltimorecity.gov/datasets/part1-crime-data/explore
crimes_2017forward = pd.read_csv ('Part1_Crime_data.csv') 

#getting rid of crimes with no date/time associated with them. Theres only 17 so not a huge loss in data
crimes_2017forward['CrimeDateTime'].isna() 
crimes_2017forward = crimes_2017forward[crimes_2017forward['CrimeDateTime'] .notna()] 

#adjusting 2017forward df so I can append the two dfs together. Involves splitting out the date and time to seperate columns, removing rows not included in both (not important), and renaming columns
for i in crimes_2017forward.index:
   crimes_2017forward.at[i, 'CrimeDate'] = crimes_2017forward.at[i,'CrimeDateTime'][5:8] + crimes_2017forward.at[i,'CrimeDateTime'][8:10] + '/' + crimes_2017forward.at[i,'CrimeDateTime'][:4]
   crimes_2017forward.at[i,'CrimeTime'] = crimes_2017forward.at[i,'CrimeDateTime'][11:19]

crimes_2017forward.drop(['X','Y', 'RowID', 'Latitude', 'Longitude', 'Premise','VRIName','Shape','CrimeDateTime' ], axis = 1, inplace = True)
crimes_2017forward.rename(columns = {'Inside_Outside': 'Inside/Outside', 'GeoLocation': 'Location 1', 'Total_Incidents':'Total Incidents'}, inplace = True)

#creatting combined crimes df from 2011 to Jan 2022
crimes = crimes_2016Novprior.append(crimes_2017forward, ignore_index = True)

#dropping unneeded column
crimes.drop(['Total Incidents'], axis = 1, inplace = True)

#del crimes_2017forward
#del crimes_2016Novprior


#--------------------------------
#Baltimore city data
#--------------------------------

#poverty by age and gender - https://datausa.io/profile/geo/baltimore-city-md#economy
#removing all the data not needed and cleaning up to summarize by year
poverty = pd.read_csv('Poverty by Age and Gender.csv')
poverty = poverty.groupby('Year').sum('Poverty Population')
poverty = poverty[['Poverty Population']]
poverty.reset_index(inplace = True)
poverty.columns = ['year','poverty_population']

#creating new dataframe consisting of annual data for the city of Baltimore
years = list(range(2011,2022))
population=[620410,622895,622391,623587,622522,616226,610481,602443,593490,584537,575584] #https://worldpopulationreview.com/us-cities/baltimore-md-population
baltimore_median_household_income = [0,0,42266,42665,44165,47350,47131,51000,50177,0,0] #https://datausa.io/profile/geo/baltimore-city-md#economy
us_median_household_income = [0,0,52250,53657,55775,57617,60336,61937,65712,0,0] #https://datausa.io/profile/geo/baltimore-city-md#economy
unemployment = [10.6,10.2,9.7,8.6,7.4,6.4,6.1,5.6,5.0,8.8] #https://msa.maryland.gov/msa/mdmanual/01glance/economy/html/unemployrates.html

baltimore_city_data = pd.DataFrame(list(zip(years, population, baltimore_median_household_income, us_median_household_income, unemployment )), columns = ['year', 'population','bal_median_income','us_median_income','unemployment_rate'])

#adding in poverty data into this dataframe and creating a poverty % of total population
baltimore_city_data = pd.merge(baltimore_city_data, poverty, how= 'left')
for i in range(2,9):
    baltimore_city_data.at[i, 'poverty_percent'] = baltimore_city_data.at[i,'poverty_population']/ baltimore_city_data.at[i,'population'] * 100

#deleting data no longer needed
del poverty
del years
del population
del baltimore_median_household_income
del us_median_household_income 
del unemployment


#--------------------------------
#misc data
#--------------------------------
#data set consisting of locatoins of CCTV's in Baltimore - https://data.world/baltimore/baltimore-cctv-locations
cctv = pd.read_csv ('CCTV_Locations.csv') 
  
#data set I made consisting of the Baltimore Raven's football schedule (NFL team) from the 2011 through 2020 seasons
football_games = pd.read_csv('ravens_schedule.csv')

for i in football_games.index:
    if len(football_games.at[i,'date']) <10:
        try:
            int(football_games.at[i,'date'][0:2]) 
            
        except ValueError:
            #football_games.at[i,'date'] = '0' +football_games.at[i,'date']
               football_games.at[i,'date'] = '0' +football_games.at[i,'date']
               if len(football_games.at[i,'date']) <10:
                   football_games.at[i,'date'] = football_games.at[i,'date'][:3] + '0' +football_games.at[i,'date'][3:]
                
        else:
            #if int(football_games.at[i,'date'][3]) < 10:
            football_games.at[i,'date'] = football_games.at[i,'date'][:3] + '0' +football_games.at[i,'date'][3:]

football_games.rename(columns = {'date':'CrimeDate'}, inplace = True) #changing date name to agree to crime df so easier to merge in
    
#--------------------------------
#adding in month, year, day of month, hour of day, week of year, and weekday
#--------------------------------

crimes['date_time_type'] = pd.to_datetime(crimes['CrimeDate']) 

for i in crimes.index:
    crimes.at[i, 'month'] = int(crimes.at[i,'CrimeDate'][:2])
    crimes.at[i, 'year'] = int(crimes.at[i,'CrimeDate'][6:])
    crimes.at[i,'day_of_month'] = int(crimes.at[i,'CrimeDate'][3:5])
    crimes.at[i,'hour_of_day'] = int(crimes.at[i,'CrimeTime'][:2])
    
crimes['week_of_year'] = crimes['date_time_type'].dt.week 
crimes['weekday'] = crimes['date_time_type'].dt.weekday #0 is Monday and 6 is Sunday



#--------------------------------
#combining all data into one
#--------------------------------

#adding in football game data. Will miss 6 games ass missing crime data from middle of Nov 2016 through end of the year    
#crimes = pd.merge(crimes, football_games, left_on = 'CrimeDate', right_on = 'date', how = 'left')
crimes = pd.merge(crimes, football_games, how = 'left')

#adding in baltimore city data
crimes = pd.merge(crimes, baltimore_city_data, how = 'outer')


del football_games


#####################################################
#Data Cleanning
#####################################################

#As we combined to historical data sets together, there is going to be some work to ensure there category's are consistent accross one another.


#--------------------------------
#CrimeCode
#--------------------------------
#will need to adjust for these as 7 or so different with a lot of actual crimes put to them
#crime code differs by - 2016 has {'1F', '4F'} & 2017 has {'1A', '3C', '3E', '3J', '3L', '5G'}
#will need to adjust these so consistent 
set(crimes_2016Novprior['CrimeCode'].unique())-set(crimes_2017forward['CrimeCode'].unique())
set(crimes_2017forward['CrimeCode'].unique()) - set(crimes_2016Novprior['CrimeCode'].unique())
set(crimes_2016Novprior['CrimeCode'].unique()).symmetric_difference(set(crimes_2017forward['CrimeCode'].unique()))

#we can now see there there is a significant amount of these transactions so should reclass them as good as possible
for i in set(crimes_2016Novprior['CrimeCode'].unique()).symmetric_difference(set(crimes_2017forward['CrimeCode'].unique())):
    print ('Total Occurences of %s' %i + ': ' + str(crimes.CrimeCode.value_counts()[i]))

#creating a list of all crime codes in data set to see which ones are similar to switch these to
crime_codes = crimes[['CrimeCode', 'Description','Weapon']]
crime_codes.drop_duplicates(inplace= True)

crime_codes2016 = crimes_2016Novprior[['CrimeCode', 'Description','Weapon']]
crime_codes2016.drop_duplicates(inplace= True)

crime_codes2017 = crimes_2017forward[['CrimeCode', 'Description','Weapon']]
crime_codes2017.drop_duplicates(inplace= True)


#reassigning codes based on the crime descriptions above. All codes are reassigned to agree to 2016 as there is less crime codes at this time. We can't adjust less into more, but can adjust more into less
#1A - 1F - This is the same as 1F in the 2016 set - Homicide with firearm

#1F - addressed in point above

#3C - 3D - This is the same as 3D in the 2016 Set - Robbery - Commercial - nan. Only 4 of these
#Per review of Baltimore crime codes 3C & 3D are comparable (both ROBB COMM) https://data.baltimorecity.gov/documents/crime-codes/about

#3E - 3F - This is the same as 3F in the 2016 Set - Robbery - Commercial - nan. Only 4 of these
#Per review of Baltimore crime codes 3E & 3F are comparable (both ROBB GAS STATION) https://data.baltimorecity.gov/documents/crime-codes/about

#3J - 3K - This is the same as 3F in the 2016 Set - Robbery - Residence - nan. Only 192 of these
#Per review of Baltimore crime codes 3E & 3F are comparable (both ROBB RESIDENCE) https://data.baltimorecity.gov/documents/crime-codes/about

#3L - 3M - This is the same as 3F in the 2016 Set - Robbery - Commercial - nan. Only 1 of these
#Per review of Baltimore crime codes 3E & 3F are comparable (both ROBB BANK) https://data.baltimorecity.gov/documents/crime-codes/about

#4F - 4E - These fall under the same crime status (Assualt by threat and Common Assualt). Looks like in the 2017 set it is no longer called Assualt by Threat so adjusted it to be common assualt

#5G - 5D - As there is only 18 of these just put them to the buurgulary category with the most. 

crimes['CrimeCode'] = crimes['CrimeCode'].replace(['1A','3C','3E','3J','3L','4F','5G'],['1F','3D','3F','3K','3M','4E','5D'])

#reviewing the final crime_codes list we can also see that there is a difference in how ARSON is treated between the two sets.
#Arson in 2016 has weapon as NA where as 2017 it has it as fire. Changed all of them to have it as NA as fire is only included as a weapon ARSON so feel like its a un needed descriptor
crimes['Weapon'] = crimes['Weapon'].replace(['FIRE'],[np.NaN])
crime_codes2017['Weapon'] = crime_codes2017['Weapon'].replace(['FIRE'],[np.NaN])

crimes.CrimeCode.isna().sum()

del crime_codes2016
del crime_codes2017
del crime_codes


#--------------------------------
#Description
#--------------------------------
#only assualt by threat which happens to have 3.5k (included in code differences above as 4F)
set(crimes_2016Novprior['Description'].unique()).symmetric_difference(set(crimes_2017forward['Description'].unique()))
crimes['Description'] = crimes['Description'].replace(['ASSAULT BY THREAT'],['COMMON ASSAULT'])

crimes.Description.isna().sum()

#--------------------------------
#Inside/Outside
#--------------------------------
#two issues to deal with here - there is some with NA and also have to rename as naming convention not consistent (Ex. some I for inside where others are Inside)
#there is 57934 NA's in total
crimes['Inside/Outside'].value_counts()
crimes['Inside/Outside'].isna().sum()

#changing naming convetion
crimes['Inside/Outside'] = crimes['Inside/Outside'].replace(['Inside', 'Outside'], ['I','O'])

#addressing NA's. Can try to use crime description to see if that can narrow it down
#doesn't appear to be any trend here with the descriptions. None of them are significant enough to draw a conclusion. 
#will leave as NA for now and comeback later if need be
na_list = crimes[crimes['Inside/Outside'].isna()]
na_list['Description'].value_counts()
na_list = na_list.Description.unique()

for i in na_list:
    temp = crimes[crimes['Description'] == i]
    print('Crime - %s \n'%i ,temp['Inside/Outside'].value_counts(),'\n')
    
del na_list

#--------------------------------
#Weapon
#--------------------------------
#only one to address here Fire, howeverthis was already addressed above under CrimeCodes and changed to NA
set(crimes_2016Novprior['Weapon'].unique()).symmetric_difference(set(crimes_2017forward['Weapon'].unique()))

#appears most crimes don't have a weapon noted on them
crimes.Weapon.isna().sum()

#baded on the crime list it seems likely that these would not of had a weapon, and as such this field was left blank
#however given weapon isn't probably to important for the end data analysis, and this assumption would be filling in 357k crimes, decided to leave as is and remove weapons from the end analysis
temp = crimes[crimes['Weapon'].isna()]
temp.Description.value_counts()

#--------------------------------
#Post
#--------------------------------
#817 posts are missing
crimes.Post.isna().sum()

#per review of Posts there doesn't appear to be enough correlation to fill in the missing posts. These data points will have to either be deleted or posts in general should be deleted. 
#leanning towards the latter as I feel like district/neighborhood will cover this anyways
temp = crimes[['Post','District', 'Neighborhood']]
temp = temp.drop_duplicates()
temp['Post'] = temp['Post'].astype(str)


#--------------------------------
#District
#--------------------------------
#name conventions slighly off and some spelling mistakes ina few, so just rename them. Pretty straight forward.
#There is one called 'Gay Street', per a quick google maps review this is downtown so changed to Central (other Central addresses appear to be downtown as well)
set(crimes_2016Novprior['District'].unique()).symmetric_difference(set(crimes_2017forward['District'].unique()))
crimes['District'] = crimes['District'].replace(['NORTHEASTERN', 'NORTHESTERN','NORTHWESTERN', 'SOUTHEASTERN', 'SOUTHESTERN','SOUTHWESTERN','Central','Gay Street'],
                                                ['NORTHEAST','NORTHEAST', 'NORTHWEST', 'SOUTHEAST', 'SOUTHEAST',  'SOUTHWEST', 'CENTRAL','CENTRAL' ])

#there is 684 NA's. Will try and reallocate based on neighborhood or street location. See more work done later. Need to clean up part of neighborhood first as this will be used to help allocate
crimes.District.isna().sum()


#--------------------------------
#Neighborhood
#--------------------------------
#At initial glance the different data sets had different formats (capitilizing letters)
#adjust all to lower case then compare to see if any differences
#only differnces left is northeastern and eastern - only 1 of each so can prob just delete
crimes_2016Novprior['Neighborhood'] = crimes_2016Novprior['Neighborhood'].str.lower()
crimes_2017forward['Neighborhood'] = crimes_2017forward['Neighborhood'].str.lower()
set(crimes_2016Novprior['Neighborhood'].unique()).symmetric_difference(set(crimes_2017forward['Neighborhood'].unique()))

#adjusting so the crimes df (final one used) has neighborhood adjusted to lower case
crimes['Neighborhood']= crimes['Neighborhood'].str.lower()

#there is 2337 missing values - to be addressed below with districts
crimes.Neighborhood.isna().sum()


#--------------------------------
#Location
#--------------------------------

#converting all locations to lower case to avoid any potential case issues
crimes['Location']= crimes['Location'].str.lower()

#there is 2723 missing values - won't be able to figure these out as it is to granular and don't have any information to figure this out
crimes.Location.isna().sum()

#to make data more usable got rid of house numbers and only kept street name. Will provide better information to the final model as more matches between locations now
for i in crimes.index:
    try:
        int(crimes.at[i,'Location'][0]) > -1
    except:
        pass
    else:
        try:
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
        except:
            pass
        
#Additional formating needed - must eliminate the following
#&
#N E W S 
#1 & 2
#1/2
#/
#001s, 04, 107.... etc ones starting with numbers still
#-fwd
#+


for i in crimes.index:
    if crimes.at[i,'Location'] == crimes.at[i,'Location']: #checks to ensure its not a NA
        if crimes.at[i,'Location'][0] == '&' and crimes.at[i,'Location'][1] != ' ':
            crimes.at[i,'Location'] = crimes.at[i,'Location'][1:]
            
        if crimes.at[i,'Location'][0] == '&' or crimes.at[i,'Location'][0] == '/':
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
            
       
        if crimes.at[i,'Location'][0:5] == '1 & 2':
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split('1 & 2 ',1)[1]
           
        if crimes.at[i,'Location'][0:3] == '1/2':
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split('1/2 ',1)[1]   
            
        if crimes.at[i,'Location'][0] == ' ':
            crimes.at[i,'Location'] = crimes.at[i,'Location'][1:]
            
        if crimes.at[i,'Location'][0:2] == 'n ' or crimes.at[i,'Location'][0:2] == 's 'or crimes.at[i,'Location'][0:2] == 'e 'or crimes.at[i,'Location'][0:2] == 'w ':
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
        
        if crimes.at[i,'Location'][0] == '/':
            crimes.at[i,'Location'] = crimes.at[i,'Location'][1:]
            
        if crimes.at[i,'Location'][0] == '+':
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
            
        if crimes.at[i,'Location'][0:4] == '-fwd':
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split('-fwd ',1)[1]
        
        if crimes.at[i,'Location'][0] == ' ':
            crimes.at[i,'Location'] = crimes.at[i,'Location'][1:]
            
        if crimes.at[i,'Location'][0:4] == '001s':  
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
            
        if crimes.at[i,'Location'][0:4] == '08 s'  or crimes.at[i,'Location'][0:4] == '09 n':  
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',2)[2]
            
        if crimes.at[i,'Location'][0:2] == '04'  or crimes.at[i,'Location'][0:2] == '09':  
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
            
        if crimes.at[i,'Location'][0:2] == '1 ':  
            crimes.at[i,'Location'] = crimes.at[i,'Location'].split(' ',1)[1]
            
        if crimes.at[i,'Location'][0:4] == '107 ':  
            crimes.at[i,'Location'] = crimes.at[i,'Location'][4:]
            


#after reviewing this it still isn't perfect as there still appears to be some errors (10k unique addresses still)
#however the first 500 locations do make up 343k of the total 500k, so we will leave it as is for now and address later if needs to be fine tuned
crimes.Location.value_counts()




#--------------------------------
#District & Neighborhood & Location - NA's
#--------------------------------
area_list = crimes[['District','Neighborhood','Location']]
area_list = area_list.drop_duplicates()

#-------- District -----------

na_district = area_list[area_list['District'].isna()]

#two of the NA districts have Neighborhoods so will use that to populate the missing values 
#neighborhoods should be assigned to the following districit per review of all the district/Neighborhood compinations
# brooklyn - SOUTHERN
# hawkins point - SOUTHERN
district_to_neighborhood = crimes[['District','Neighborhood']]
district_to_neighborhood = district_to_neighborhood.drop_duplicates()


#looking into at a location level to assign districts since missing neighborhoods
district_to_location = crimes[['District','Location']]
district_to_location = district_to_location.drop_duplicates()
district_to_location = district_to_location[district_to_location['District'].notna()]



for i in na_district.index:
    try:
        temp = district_to_location[district_to_location['Location'] == na_district.at[i,'Location']].reset_index()
        na_district.at[i,'District'] = temp.at[0, 'District']
    except:
        pass

#only 56 left NA, can leave with this for now
na_district['District'].isna().sum()


#-------- Neighborhood --------


na_neighborhood = area_list[area_list['Neighborhood'].isna()]

#looking into at a location level to assign districts since missing neighborhoods
neighborhood_to_location = crimes[['Neighborhood','Location']]
neighborhood_to_location = neighborhood_to_location.drop_duplicates()
neighborhood_to_location = neighborhood_to_location[neighborhood_to_location['Neighborhood'].notna()]

for i in na_neighborhood.index:
    try:
        temp = neighborhood_to_location[neighborhood_to_location['Location'] == na_neighborhood.at[i,'Location']].reset_index()
        na_neighborhood.at[i,'Neighborhood'] = temp.at[0, 'Neighborhood']
    except:
        pass

#only 73 left NA, can leave with this for now
na_neighborhood['Neighborhood'].isna().sum()

#------ Populating Crimes DF ----

for i in na_district.index:
    crimes.at[i,'District'] =  na_district.at[i,'District']

for i in na_neighborhood.index:
    crimes.at[i,'Neighborhood'] =  na_neighborhood.at[i,'Neighborhood']

#small decrease in number of NA's, can probably get it down lower with more cleaning to the locations but left as is for now as it will be a lot more work
crimes.District.isna().sum() #570 - down from 684
crimes.Neighborhood.isna().sum() #2190 - down from 2337

del crimes_2016Novprior
del crimes_2017forward
del district_to_location
del district_to_neighborhood
del na_district
del na_neighborhood
del neighborhood_to_location
del temp
del area_list



quit()


#####################################################
#Explority Data Analysis
#####################################################

#now that the data set is cleaned up, we can do some EDA over it to start to draw some conclusions

#--------------------------------
#CrimeCode
#--------------------------------

#you can definetly see there is a few crime codes that are the most common, particularily the top 4
crimes.CrimeCode.value_counts().iloc[:15].sort_values().plot(kind="barh", title = "Crimes by Crime Code") #shows top 15

#the top 4 crime codes cover 48% of all the crimes that occured
#this is impressive given there is 80 crime codes in all, so 5% of the crime codes almost covers 50% of the crimes
crimes.CrimeCode.value_counts().iloc[:4].sum()  / len(crimes)
crimes.CrimeCode.nunique()


#--------------------------------
#Description
#--------------------------------

#again we can see there is large about 7 crimes that make up the majority. 
crimes.Description.value_counts().sort_values().plot(kind="barh", title = "Crimes by Description") #shows top 15

#the top 3 crimes cover 54% of all the crimes that occrued
#top 5 covers 78%
crimes.Description.value_counts().iloc[:3].sum()  / len(crimes)
crimes.Description.value_counts().iloc[:5].sum()  / len(crimes)


#--------------------------------
#Inside/Outside
#--------------------------------

#we can see now that it is almost a 50/50 split between inside vs outside crimes - 43% inside 57% outside
#as this is close to 50/50, this data point may not provide much beneficial information. To see if there is lets look at crime descriptions to see if a correlation between them being inside or outside
crimes['Inside/Outside'].value_counts().sort_values().plot(kind="barh", title = "Crimes by Inside/Outside") 
len(crimes[crimes['Inside/Outside'] == 'I']) / len(crimes)

#There does appear to be correlatoin between crime inside vs outisde for certain crimes - especially auto related crimes
temp = crimes
temp = temp[temp['Inside/Outside'].notna()]
temp = temp.groupby(['Description','Inside/Outside']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['Occurences']
temp.reset_index(inplace = True)
temp.sort_values(by = ['Description'], inplace = True)

temp1 = temp[temp['Inside/Outside'] == 'I']

temp2 = temp[temp['Inside/Outside'] == 'O']

plt.bar(temp1['Description'], temp1['Occurences'], label = 'Inside')
plt.bar(temp2['Description'], temp2['Occurences'], bottom = temp1['Occurences'] , label = 'Outside')
plt.title('Crimes per Inside vs Outside')
plt.xlabel('Crime Occurences')
plt.ylabel('Crime')
plt.xticks(rotation = 90)
plt.legend()

del temp1
del temp2

#getting rid of auto related crimes flips it almost opposite of before - 54% inside now and 46% outside
temp = crimes[crimes['Description'] != 'AUTO THEFT']
temp = temp[temp['Description'] != 'LARCENY FROM AUTO']
temp = temp[temp['Description'] != 'ROBBERY - CARJACKING']

len(temp[temp['Inside/Outside'] == 'I']) / len(temp)


#--------------------------------
#Weapon
#--------------------------------

#as noted above weapons is removed from the end analysis due to there being to many NaN valeus (357k of 500K total)
#this is to large to try and make an assumption, so left out


#--------------------------------
#Post
#--------------------------------

#as noted above posts is likely going to get removed from the end DF so no EDA done over it


#--------------------------------
#District
#--------------------------------

#as there is 570 NAs, will remove these values when doing the EDA over it
crimes.District.isna().sum()

#based on this it appears the east side of the city has more crime then the west
crimes[crimes['District'].notna()] ['District'].value_counts().sort_values().plot(kind="barh", title = "Crimes by District") 


#--------------------------------
#Neighborhood
#--------------------------------

#as there is 2190 NAs, will remove these values when doing the EDA over it
crimes.Neighborhood.isna().sum()

#crime falls off in certain neighborhoods. Downtown appears to have the highest crime. Nothing further to dig into yet
crimes[crimes['Neighborhood'].notna()] ['Neighborhood'].value_counts().iloc[:].sort_values().plot(kind="barh", title = "Crimes by Neighborhood") 
crimes[crimes['Neighborhood'].notna()] ['Neighborhood'].value_counts().iloc[:25].sort_values().plot(kind="barh", title = "Crimes by Neighborhood") 


#--------------------------------
#month
#--------------------------------

#no NA to worry about
crimes.month.isna().sum()

#from these we can see that the crime appears to be slowest from Nov to April, afterwards it reaches its high from May to Oct..
#from these we can see that the seasons affect the amount of crime- likely due to the weather being colder

plt.plot(crimes.groupby(['month']).month.count())
plt.title('Crimes by Month')
plt.xlabel('Month')
plt.ylabel('Crime Occurences')
plt.show()

#for additional analysis lets see if the amount of crimes inside vs outside is different throughout the year
#from this we can see that the decrease in winter crimes is primairly due to lower outside crimes as the inside crimes stay relatively consistent

temp = crimes
temp = temp[temp['Inside/Outside'].notna()]
temp = temp.groupby(['month','Inside/Outside']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['Occurences']
temp.reset_index(inplace = True)
temp.sort_values(by = ['month'], inplace = True)

temp1 = temp[temp['Inside/Outside'] == 'I']

temp2 = temp[temp['Inside/Outside'] == 'O']

plt.bar(temp1['month'], temp1['Occurences'], label = 'Inside')
plt.bar(temp2['month'], temp2['Occurences'], bottom = temp1['Occurences'] , label = 'Outside')
plt.title('Crimes per Inside vs Outside')
plt.xlabel('Crime Occurences')
plt.ylabel('Crime')
plt.xticks(rotation = 90)
plt.legend()

del temp1
del temp2

#as outside crimes are lower in winter we would expect a decrease in the crimes that are more commonly outside during the winter (auto related crimes as noted earlier)
#however looking at the chart it doesn't appear to be as durastic as would of orignaly thought
temp = crimes.groupby(['Description','month']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['Description'].unique():
    plt.plot(
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['month'],
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Month Per Description')
    plt.xlabel('Month')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['month'].unique())
    plt.legend()
    
  
    
#--------------------------------
#year
#--------------------------------

#no NA to worry about
crimes.year.isna().sum()

#we can see that crime took a large decrease in 2016 however it spiked up hard in 2017 - up due to Trump taking office? outrage?. I looked up most notable news from Baltimore in 2017 and nothing was to large to warrant this increase
#We can also see come 2020 and 2021 crime decreased significantly compared to the normal - this is likely due to COVID and people staying in more
#Also note the large drop in 2022 - this is do to 2022 only including partial data right now
plt.plot(crimes.groupby(['year']).year.count())
plt.title('Crimes by Year')
plt.xlabel('Year')
plt.ylabel('Crime Occurences')
plt.show()


#lets compare the crimes by month per year to ensure the same seasonaility exists
#2020 and 2021 look strange, again given COVID. These years should probably be removed from the final model
#Everything else looks relatively usual other then Jan 2017 being significantly higher then normal. 
#This could be as this is when Trump took office? Makes sense as Maryland tends to be a strong Democratic state
temp = crimes.groupby(['month','year']).year.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['year'].unique():
    plt.plot(
        ((temp[temp['year'] == i]).drop(labels = 'year', axis = 1))['month'],
        ((temp[temp['year'] == i]).drop(labels = 'year', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Month Per Year')
    plt.xlabel('Month')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['month'].unique())
    plt.legend()

#lastly lets check to see if the crime types have become more or less common - any trend
#nothing to notable up till 2020 where it fell off due to COVID
temp = crimes.groupby(['Description','year']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['Description'].unique():
    plt.plot(
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['year'],
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Year Per Description')
    plt.xlabel('Year')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['year'].unique())
    plt.legend()



#--------------------------------
#day_of_month
#--------------------------------

#no NA to worry about
crimes.day_of_month.isna().sum()

#initially we can see there is a sharp decrease after the 28th. This would be expected as February will now not been included 
#we can see that the highest crime is at the start of the month, otherwise it tends to hold relatively stable.
plt.plot(crimes.groupby(['day_of_month']).day_of_month.count())
plt.title('Crimes per Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Crime Occurences')
plt.xticks(range(2,31,2))
plt.show()



#--------------------------------
#hour_of_day
#--------------------------------

#appears to be 1 crime that occured on the 24 hour which shouldnt happen as the clock stops at 23:59
crimes['hour_of_day'].value_counts() 
crimes.index[crimes['hour_of_day'] == 24].tolist() #apperas to be in index 198664, going to compare to other indexes before and after to see try and decide where it should go
crimes[198660:198668] #times appear to follow somewhat chronologically with the index. As such looking at these rows, as this time is dated for Sept 24 and the other times for Sept 24 are towards midnight, this time must also of been at midnight on the 24th. As such adjusted it to be 23 hours.
crimes['hour_of_day'][198664] = 23
crimes['hour_of_day'][198661] = 23 #?????? not sure why this one is showing up as well as 24
sorted(crimes['hour_of_day'].unique()) #all appears fixed now

#no NA to worry about
crimes.hour_of_day.isna().sum()

#crime appears to fall off after midnight for most of the night and doesn't start picking up until people start going to work
#it hits its highs from the afternoon through the rest of the day 
plt.plot(crimes.groupby(['hour_of_day']).hour_of_day.count())
plt.title('Crimes per Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Crime Occurences')

plt.show()

#looking at this we can definetly notice that certain crimes occur more depending on the time of the day.
#This is obvious with Larceny where crimes occur almost double from the hours of 11 to 17 copmared to any other point of the day. (people are at work so there not around?)
temp = crimes.groupby(['Description','hour_of_day']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['Description'].unique():
    plt.plot(
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['hour_of_day'],
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Hour of Day Per Description')
    plt.xlabel('Hour of Day')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['hour_of_day'].unique())
    #plt.legend()
    
    
#looking at this we can see that the crimes follow the same pattern throughout the hours of day regardless of season or month
temp = crimes.groupby(['month','hour_of_day']).year.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['month'].unique():
    plt.plot(
        ((temp[temp['month'] == i]).drop(labels = 'month', axis = 1))['hour_of_day'],
        ((temp[temp['month'] == i]).drop(labels = 'month', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Hour of Day Per Month')
    plt.xlabel('Hour of Day')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['hour_of_day'].unique())
    plt.legend()


#--------------------------------
#week_of_year
#--------------------------------

#no NA to worry about
crimes.week_of_year.isna().sum()

crimes['week_of_year'].value_counts() 
temp = crimes[crimes['week_of_year'] == 53]


#Doesn't appear to be any trend that is to apprent, just the same seasonality as noted above when looking at months. 
#Is a bit more obvious and severe drop off in crime here when winter first comes around towards the end of the year
plt.plot(crimes.groupby(['week_of_year']).week_of_year.count())
plt.title('Crimes per Week of Year')
plt.xlabel('Week of Year')
plt.ylabel('Crime Occurences')
plt.xticks(range(3,52,4))
plt.show()



#--------------------------------
#weekday
#--------------------------------

#no NA to worry about
crimes.weekday.isna().sum()

#ensuring weekdays are reasonable
#check total sum of weekday int to expected days for the 10 year period - 52 weeks plus 10 days to cover the 10 years + 2 leap days. For additional days used the average of 3 for the weekday int
#diff of -3 is insignificant, apperas reasonable 
temp = crimes[['weekday','year','CrimeDate']]
temp.drop_duplicates(inplace = True)
temp = temp[temp['year'] != 2016] 
temp = temp[temp['year'] != 2022] 
temp['weekday'].sum() - ( (0+1+2+3+4+5+6) * 52 * 10 + (3 * 12) ) 


#we can see crime is the highest on Friday bya large margin and falls off on the weekend, with Sunday being by far athe lowest
plt.plot(crimes.groupby(['weekday']).weekday.count())
plt.title('Crimes per Weekday')
plt.xlabel('Weekday (0 = Monday)')
plt.ylabel('Crime Occurences')
plt.xticks(crimes['weekday'].unique())
plt.show()

#lets look to see if certain crimes occur more on weekend or weekday.
#pretty minimal change. Only notable ones is Larceny and Burglary which fall off strongly on the weekend. 
#As these are high occuring crimes, it makes sense that this would push the total trend to fall off strong on weekends
temp = crimes.groupby(['Description','weekday']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['Description'].unique():
    plt.plot(
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['weekday'],
        ((temp[temp['Description'] == i]).drop(labels = 'Description', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Weekday Per Description')
    plt.xlabel('Weekday')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['weekday'].unique())
    #plt.legend()


#looking at each districts crime breakdown per weekday we can see that some districts dont follow the pattern as closely
#most notabely southeast and southern
temp = crimes.groupby(['District','weekday']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['District'].unique():
    plt.plot(
        ((temp[temp['District'] == i]).drop(labels = 'District', axis = 1))['weekday'],
        ((temp[temp['District'] == i]).drop(labels = 'District', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Weekday Per District')
    plt.xlabel('Weekday')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['weekday'].unique())
    #plt.legend()




#--------------------------------
#football games
#--------------------------------
#belief here is that football games bring lots of people together and drinking/getting rowdy. As a result would expect a increase in crime on days of football games

#we can initially see that if the game is home or away it has little to no affect on the amount of crimes that day
#as such we are only concerned if football is in sesaon or not
temp = crimes[crimes['home/away'].notna()]
temp['home/away'].value_counts()



#looking at crimes per weekday per month, if football had a correlation with crime we would expect...
#a increase or less drop off form saturday, in the amount of crimes on Sunday through football season (Sept to Dec/Jan). 
temp = crimes.groupby(['month','weekday']).Description.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

for i in crimes['month'].unique():
    plt.plot(
        ((temp[temp['month'] == i]).drop(labels = 'month', axis = 1))['weekday'],
        ((temp[temp['month'] == i]).drop(labels = 'month', axis = 1))['occurences'],
        label = 'Crime %s' % i)
    plt.title('Crimes per Weekday Per Month')
    plt.xlabel('Weekday')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['weekday'].unique())
    plt.legend()


#as the stadium is downtow we'd expect the crime to be higher downtown during home games
#however this doesn't appear true based on below.
temp = crimes[crimes['home/away'].notna()]
temp = temp[temp['District'] == 'CENTRAL']
temp['home/away'].value_counts()

#I believe its safe to say there is no correlation between criem and football at this point


#--------------------------------
#population
#--------------------------------

#adding in crimes occured into the baltimore_city_data
temp = crimes.groupby(['year']).year.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)
baltimore_city_data = pd.merge(baltimore_city_data, temp)

baltimore_city_data['crime_per_capita'] = baltimore_city_data['occurences'] /baltimore_city_data['population'] * 100

#from below we can see that hte population has been decreasing steadily since 2015 but we have not seen that same decrease in crimes
#we can see from the crime per capita, that it actually increased in 2017-2019 from where it was in 2015.
plt.plot(baltimore_city_data['year'], baltimore_city_data['population'])
plt.title('Population by Year')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()

plt.plot(crimes.groupby(['year']).year.count())
plt.title('Crimes by Year')
plt.xlabel('Year')
plt.ylabel('Crime Occurences')
plt.show()

plt.plot(baltimore_city_data['year'], baltimore_city_data['crime_per_capita'])
plt.title('Crime per Capita by Year')
plt.xlabel('Year')
plt.ylabel('Crime per Capita')
plt.show()




#--------------------------------
#poverty_population & poverty_percent
#--------------------------------

#again we see no correlation. The percentage of the population under the poverty threshold has decreased each  year since 2014 yet we haven't seen a similar change in crimes
#this is a bit surprising
plt.plot(baltimore_city_data['year'], baltimore_city_data['poverty_percent'])
plt.title('Poverty Percent by Year')
plt.xlabel('Year')
plt.ylabel('Poverty Percent')
plt.show()

plt.plot(crimes.groupby(['year']).year.count())
plt.title('Crimes by Year')
plt.xlabel('Year')
plt.ylabel('Crime Occurences')
plt.show()




#--------------------------------
#bal_median_income & us_median_income
#--------------------------------

#the thought here is this is a picture of the overall economy, so an improved economy should have less crime
#from below we can see that firstly the income difference has slowly been climbining as have both the incomes (US and Baltimore)
#the difference doesn't seem to correlate much with the crime other then the spike in the large median difference in 2017 and the crime in 2017
#but in 2019 the median income difference is the highest, but at this point crime is dropping Baltimore. So tough to say if theres a conclusive argument here

baltimore_city_data['median_income_diff'] = baltimore_city_data['us_median_income'] - baltimore_city_data['bal_median_income']

#removing years with no data 
temp = baltimore_city_data[baltimore_city_data['year'] >2012]
temp = temp[temp['year'] <2020]

plt.plot(temp['year'], temp['bal_median_income'], label = 'Baltimore')
plt.plot(temp['year'], temp['us_median_income'], label = 'US')
plt.plot(temp['year'], temp['median_income_diff'], label = 'Difference')
plt.title('US vs Baltimore Median Household Income per Year')
plt.xlabel('Year')
plt.ylabel('Median Household Income')
plt.legend()
plt.show()


plt.plot(crimes.groupby(['year']).year.count())
plt.title('Crimes by Year')
plt.xlabel('Year')
plt.ylabel('Crime Occurences')
plt.show()

#--------------------------------
#unemployment_rate
#--------------------------------

#notable thing here is the huge increase in unemployment in 2020. This would be expected as a result of COVID
#There does appear to be a small correlation here. Large decrease in unemployment rate from 2011 to 2015
#We can see the crime slowly decrease over these years - very slowly though
plt.plot(baltimore_city_data['year'], baltimore_city_data['unemployment_rate'])
plt.title('Unemployment Rate by Year')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.show()


plt.plot(crimes.groupby(['year']).year.count())
plt.title('Crimes by Year')
plt.xlabel('Year')
plt.ylabel('Crime Occurences')
plt.show()




#####################################################
#Modeling
#####################################################

#--------------------------------
#data clean up
#--------------------------------


temp = crimes[crimes['District'].notna()]
temp = crimes[crimes['Neighborhood'].notna()]
columns_to_drop = ['Inside/Outside', 'Weapon','season', 'regular/playoff', 'home/away', 'win/loss', 'Location 1', 
                   'CrimeDate', 'CrimeTime', 'Location','Post', 'date_time_type', 'week_of_year']
temp.drop(columns_to_drop, axis = 1, inplace = True)

temp['median_income_diff'] = temp['us_median_income'] - temp['bal_median_income']
#????????? temp['crime_per_capita'] = temp['occurences'] /temp['population'] * 100 could add as a lag?

#bunch to eliminate. Got rid of. Should either try and populate those years or just eliminate those years and reduce data pool size
temp.isna().sum()
columns_to_drop = ['population', 'bal_median_income', 'us_median_income', 'unemployment_rate', 'poverty_population',
                   'poverty_percent', 'median_income_diff']
temp.drop(columns_to_drop, axis = 1, inplace = True)

temp = temp.astype({"Neighborhood":'category', "District":'category'})
temp['Neighborhood'] = LabelEncoder().fit_transform( temp.Neighborhood )
temp['District'] = LabelEncoder().fit_transform( temp.District )


#--------------------------------
#model on the month level
#--------------------------------
temp = temp.groupby(['month','year', 'Neighborhood', 'District']).year.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)
temp = temp[temp['year']<2020]

#average crimes per neighborhood
temp['occurences'].mean() 
#model produces a MAE of 1.16 where as the total average crimes is 13.4 - not bad?


from xgboost import XGBRegressor
from matplotlib.pylab import rcParams

data = temp.copy()
X_train = data[(data.year <= 2019) & (data.month < 11)].drop(['occurences'], axis=1)
Y_train = data[(data.year <= 2019) & (data.month < 11)]['occurences']
X_valid = data[(data.year >= 2019) & (data.month >= 11)].drop(['occurences'], axis=1)
Y_valid = data[(data.year >= 2019) & (data.month >= 11)]['occurences']
#X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)
del data

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1, 
    seed=42) 

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", #rmse or mae
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True)#, 
    #early_stopping_rounds = 20)



from xgboost import plot_importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
plot_features(model, (10,14))


#--------------------------------
#model on the day level
#--------------------------------

#group by day - could add in crime description as well and crime code? maybe to granular at that point
temp = temp.groupby(['month','year', 'Neighborhood', 'District', 'day_of_month']).year.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)
temp = temp[temp['year']<2020]

#average crimes per neighborhood
temp['occurences'].mean() 
#model produces a MAE of 0.47 where as the total average crimes is 1.57. This doesn't feel to good


from xgboost import XGBRegressor
from matplotlib.pylab import rcParams

data = temp.copy()
X_train = data[(data.year <= 2019) & (data.month < 11)].drop(['occurences'], axis=1)
Y_train = data[(data.year <= 2019) & (data.month < 11)]['occurences']
X_valid = data[(data.year >= 2019) & (data.month >= 11)].drop(['occurences'], axis=1)
Y_valid = data[(data.year >= 2019) & (data.month >= 11)]['occurences']
#X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
Y_train = Y_train.clip(0, 20)
Y_valid = Y_valid.clip(0, 20)
del data

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1, 
    seed=42) 

model.fit(
    X_train, 
    Y_train, 
    eval_metric="mae", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True)#, 
    #early_stopping_rounds = 20)



from xgboost import plot_importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
plot_features(model, (10,14))






