#looks like crime decreasing throughout the years - maybe do a quick research and tie to baltimore economy data? hopefully increasing or something? or is population decreasing so rate staying the same?
#should we add population in, and do a crime rate calc per year?
#will need to adjsut anything looking at days/years/months as the data for 2016 stops at Nov 12/2016 so missing a month and half
#address other missing values and outliers in data if any
#maybe look at crime by area?
#create model to predict occurences of crime based on day, hour, etc and maybe neighborhood
#tie in over arching economic data for the US? has it been doing better recently or later
#tie in with Baltimore Football games
#weather?

#????? ensure to adjust for only half of Nov 2016 and no Dec 2016
#????? will probably want to use averages for any plotting as this should solve the issue
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


quit()


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
#???????????? to look into rest
#converting all locations to lower case to avoid any potential case issues
crimes['Location']= crimes['Location'].str.lower()

#there is 2723 missing values - to be addressed below
crimes.Location.isna().sum()


#--------------------------------
#District & Neighborhood & Location - NA's
#--------------------------------

na_locations = crimes[['District', 'Neighborhood','Location']]
na_locations.drop_duplicates(inplace= True)

#498,384 of our total 507,001 appear to be streets with numbers, the remaining 8917 dont have street numbers attached
counter = 0
for i in crimes.index:
    try:
        if int(crimes.at[i,'Location'][0]) > -1:
            counter +=1
    except:
        pass
        
print(counter)
del counter




#need to clear out & and maybe s n w e 


temp = crimes[['District', 'Neighborhood','Location']]
temp2 = temp

#to make data more usable got rid of house numbers and only kept street name. Will provide better information to the final model as more matches between locations now
for i in temp.index:
    try:
        int(temp.at[i,'Location'][0]) > -1
    except:
        pass
    else:
        try:
            temp.at[i,'Location'] = temp.at[i,'Location'].split(' ',1)[1]
        except:
            pass
           
#further simplifying the locations by getting rid of any streets that start with & (some were just like this in the data set) or ones with N S E W references
#NSEW references not needed as model shouldn't care which one its on (generally they a N vs S st of same name should be fairly close)
for i in temp.index:
    try:
        if temp.at[i,'Location'][0] == '&' or temp.at[i,'Location'][0:2] == 'n ' or temp.at[i,'Location'][0:2] == 's 'or temp.at[i,'Location'][0:2] == 'e 'or temp.at[i,'Location'][0:2] == 'w ':
            temp.at[i,'Location'] = temp.at[i,'Location'].split(' ',1)[1]
    except:
        pass




i = 304660


temp.iloc[[470]]
temp2['Location'][261332]
temp['Location'][126123]
   
a = 'n 6800 BOSTON AVE'
b = ''

if a[0] == '&' or a[0:2] == 'n ' or a[0:2] == 's 'or a[0:2] == 'e 'or a[0:2] == 'w ':
    b = a.split(' ',1)[1]
    print('true')

    
    
b = a.split(' ',1)[1]













########################################################
########################################################
########################################################


crimes.CrimeCode.nunique()
crimes.CrimeCode.value_counts()
#crimes.CrimeCode.value_counts().iloc[:10].sort_values().plot(kind="barh", title = "Types of Crimes") #shows top 10


crimes.Description.nunique()
crimes.Description.value_counts()
#crimes.Description.value_counts().sort_values().plot(kind="barh", title = "Types of Crimes") #shows all

crimes['Inside/Outside'].nunique()
crimes['Inside/Outside'].value_counts()

crimes.Weapon.nunique()
crimes.Weapon.value_counts()

crimes.Post.nunique()
crimes.Post.value_counts()

crimes.Neighborhood.nunique()
crimes.Neighborhood.value_counts()

crimes.District.nunique()
crimes.District.value_counts()



##############
#adding in month and year
###############

crimes['date_time_type'] = pd.to_datetime(crimes['CrimeDate']) 

for i in crimes.index:
    crimes.at[i, 'month'] = int(crimes.at[i,'CrimeDate'][:2])
    crimes.at[i, 'year'] = int(crimes.at[i,'CrimeDate'][6:])
    crimes.at[i,'day_of_month'] = int(crimes.at[i,'CrimeDate'][3:5])
    crimes.at[i,'hour_of_day'] = int(crimes.at[i,'CrimeTime'][:2])
    
crimes['week_of_year'] = crimes['date_time_type'].dt.week 
crimes['weekday'] = crimes['date_time_type'].dt.weekday #0 is Monday and 6 is Sunday


##############
#data cleaning 
###############

#hour_of_day
crimes['hour_of_day'].value_counts() #appears to be 1 crime that occured on the 24 hour which shouldnt happen as the clock stops at 23:59
crimes.index[crimes['hour_of_day'] == 24].tolist() #apperas to be in index 198664, going to compare to other indexes before and after to see try and decide where it should go
crimes[198660:198668] #times appear to follow somewhat chronologically with the index. As such looking at these rows, as this time is dated for Sept 24 and the other times for Sept 24 are towards midnight, this time must also of been at midnight on the 24th. As such adjusted it to be 23 hours.
crimes['hour_of_day'][198664] = 23
sorted(crimes['hour_of_day'].unique()) #all appears fixed now

#month
crimes['month'].value_counts() #no issues, all appears appropriate

#year
crimes['year'].value_counts() #no issues, all appears appropriate

#day_of_month
crimes['day_of_month'].value_counts() #no issues, all appears appropriate

#week_of_year
crimes['week_of_year'].value_counts() #no issues, all appears appropriate

#weekday
crimes['weekday'].value_counts()



quit()

crimes = crimes[crimes['date_time_type'] < '2016-01-01']

plt.plot(crimes.groupby(['year']).year.count())
plt.title('Crimes per Year')
plt.xlabel('Year')
plt.ylabel('Crime Occurences')
plt.xticks(crimes['year'].unique())
plt.show()

plt.plot(crimes.groupby(['month']).month.count())
plt.title('Crimes per Month')
plt.xlabel('Month')
plt.ylabel('Crime Occurences')
plt.show()

plt.plot(crimes.groupby(['day_of_month']).day_of_month.count())
plt.title('Crimes per Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Crime Occurences')
plt.show()

plt.plot(crimes.groupby(['hour_of_day']).hour_of_day.count())
plt.title('Crimes per Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Crime Occurences')
plt.show()

plt.plot(crimes.groupby(['week_of_year']).week_of_year.count())
plt.title('Crimes per Week of Year')
plt.xlabel('Week of Year')
plt.ylabel('Crime Occurences')
plt.show()


plt.plot(crimes.groupby(['weekday']).weekday.count())
plt.title('Crimes per Weekday')
plt.xlabel('Weekday (0 = Monday)')
plt.ylabel('Crime Occurences')
plt.show()


#by weekday by time
temp = crimes.groupby(['weekday','hour_of_day']).weekday.count()
temp = pd.DataFrame(temp)
temp.columns = ['occurences']
temp.reset_index(inplace = True)

#%matplotlib qt makes plot show outside of GUI
#%matplotlib

for i in range(0,7):
    plt.plot(
        ((temp[temp['weekday'] == i]).drop(labels = 'weekday', axis = 1))['hour_of_day'],
        ((temp[temp['weekday'] == i]).drop(labels = 'weekday', axis = 1))['occurences'],
        label = 'day %d' % i
        )
    plt.title('Crimes per Hour Per Weekday ( 0 = Monday)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Crime Occurences')
    plt.xticks(crimes['hour_of_day'].unique())
    plt.legend()
    #plt.show()



quit()
#crime by day of month
#crime by hour of day

crimes.year.value_counts()
crimes.month.value_counts()

############
#distances
#############

#converts all location coordinates to a tuple so it can be used in the distance calculation
#in crime data the following rows do not
crimes['Location 1'].isnull().sum() #there is 1619 of NA values in location. We could attempt to fill these in but as there is pretty few just removed them
crimes.dropna(subset=['Location 1'], inplace = True) #removed all NA's in location 1 column
crimes.reset_index(inplace = True)


for i in crimes.index:
    location = crimes['Location 1'][i]
    location = location[1:-1]
    location = location.replace(' ','')
    location = tuple(map(float, location.split(',')))
    crimes.at[i, 'Location 1'] = location


for i in cctv.index:
    location = cctv['Location 1'][i]
    location = location[1:-1]
    location = location.replace(' ','')
    location = tuple(map(float, location.split(',')))
    cctv.at[i, 'Location 1'] = location
    
quit()
    
    
#mapping

map = folium.Map(location = [39.290996, -76.621074], zoom_start = 14, control_scale=True)

crimes = crimes[crimes['Description'] == 'LARCENY']

for i in cctv.index:
     folium.Marker(
        location= cctv.at[i, 'Location 1'],
        radius=5,
        popup= cctv.at[i, 'cameraNumber'],
        color='red',
        fill=True,
        fill_color='red'
        ).add_to(map)   


for i in cctv.index:
     folium.Circle(
        location= cctv.at[i, 'Location 1'],
        radius=50,
        popup= cctv.at[i, 'cameraNumber'],
        color='black',
        fill=False,
        ).add_to(map)   
     
 

for i in crimes.index:
     folium.Circle(
        location= crimes.at[i, 'Location 1'],
        radius=2,
        #popup= cctv.at[i, 'cameraNumber'],
        color='red',
        fill=False,
        ).add_to(map)   
 

for i in cctv.index:
     folium.CircleMarker(
        location= cctv.at[i, 'Location 1'],
        radius=20,
        popup= cctv.at[i, 'cameraNumber'],
        color='blue',
        fill=True,
        fill_color = 'blue'
        ).add_to(map)  
    
map.add_child(FastMarkerCluster(crimes['Location 1'].values.tolist()))

map.add_child(HeatMap(crimes['Location 1'].values.tolist()))


map.save('map.html')
webbrowser.open('map.html')

    

#hs.haversine(loc1, loc2) #default is KM
from haversine import Unit
#hs.haversine(loc1, loc2, unit = Unit.METERS) #changes to M





