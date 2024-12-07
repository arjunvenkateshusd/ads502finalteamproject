# Predicting Student Academic Success
This project is a part of the ADS-502 course in the Applied Data Science Program at the University of San Diego. 
</p>

#### Project Status: [Completed]

## Installation
To use this project, first clone the repo using the command below: </p>
git init </p>
git clone https://github.com/arjunvenkateshusd/ads502finalteamproject.git </p>

## Project Intro/Objective 
The main purpose of this project is to identify students at risk of dropping out from higher education institutions by leveraging predictive analytics. By analyzing key features such as academic performance, gender, financial status, and socio-demographic factors, this project aims to provide actionable insights to educators and administrators. Early identification of at-risk students allows for timely interventions, such as tailored academic support, counseling, or financial aid, to improve retention rates.
</p>
The potential impact of this project includes fostering student success, reducing institutional dropout rates, and promoting equity in education by addressing challenges faced by vulnerable student populations. This application can help institutions optimize resources, enhance academic outcomes, and support the long-term success of their students.

## Partner(s)/Contributor(s)
• Akshat Patni, Arjun Venkatesh, and Cynthia Portales-Loebell

## Methods Used
• Inferential Statistics </p>
• Data Mining </p>
• Predictive Modeling </p>
• Machine Learning </p>
• Data Visualization </p>
• Programming </p>
• Data Manipulation </p>

## Technologies
• Python

## Project Description
This project aims to predict academic success in higher education institutions by identifying students at risk of dropping out. By leveraging data-driven techniques, the project seeks to provide actionable insights that can aid administrators in implementing targeted interventions to improve student retention rates. </p>
#### Dataset Details
Data Source: The dataset was sourced from Kaggle, which provides information on student demographics, academic performance, financial status, gender, and other relevant features. </p>
Number of Variables: 37 features, including categorical and numerical variables like age, grade, gender, financial aid status, and course units. </p>
Size of Dataset: 76,518 records. </p>
Data Dictionary: </p>
• Marital status: 1 – single 2 – married 3 – widower 4 – divorced 5 – facto union 6 – legally separated                                     
• Application mode: 1 - 1st phase - general contingent 2 - Ordinance No. 612/93 5 - 1st phase - special contingent (Azores Island) 7 - Holders of other higher courses 10 - Ordinance No. 854-B/99 15 - International student (bachelor) 16 - 1st phase - special contingent (Madeira Island) 17 - 2nd phase - general contingent 18 - 3rd phase - general contingent 26 - Ordinance No. 533-A/99, item b2) (Different Plan) 27 - Ordinance No. 533-A/99, item b3 (Other Institution) 39 - Over 23 years old 42 - Transfer 43 - Change of course 44 - Technological specialization diploma holders 51 - Change of institution/course 53 - Short cycle diploma holders 57 - Change of institution/course (International)                                  
• Application order: between 0 - first choice; and 9 last choice                                
• Course: 33 - Biofuel Production Technologies 171 - Animation and Multimedia Design 8014 - Social Service (evening attendance) 9003 - Agronomy 9070 - Communication Design 9085 - Veterinary Nursing 9119 - Informatics Engineering 9130 - Equinculture 9147 - Management 9238 - Social Service 9254 - Tourism 9500 - Nursing 9556 - Oral Hygiene 9670 - Advertising and Marketing Management 9773 - Journalism and Communication 9853 - Basic Education 9991 - Management (evening attendance)                                           
• Daytime/evening attendance: 1 – daytime 0 - evening                       
• Previous qualification: 1 - Secondary education 2 - Higher education - bachelor's degree 3 - Higher education - degree 4 - Higher education - master's 5 - Higher education - doctorate 6 - Frequency of higher education 9 - 12th year of schooling - not completed 10 - 11th year of schooling - not completed 12 - Other - 11th year of schooling 14 - 10th year of schooling 15 - 10th year of schooling - not completed 19 - Basic education 3rd cycle (9th/10th/11th year) or equiv. 38 - Basic education 2nd cycle (6th/7th/8th year) or equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 42 - Professional higher technical course 43 - Higher education - master (2nd cycle)                            
• Previous qualification (grade): Grade of previous qualification (between 0 and 200)                  
• Nacionality: 1 - Portuguese; 2 - German; 6 - Spanish; 11 - Italian; 13 - Dutch; 14 - English; 17 - Lithuanian; 21 - Angolan; 22 - Cape Verdean; 24 - Guinean; 25 - Mozambican; 26 - Santomean; 32 - Turkish; 41 - Brazilian; 62 - Romanian; 100 - Moldova (Republic of); 101 - Mexican; 103 - Ukrainian; 105 - Russian; 108 - Cuban; 109 - Colombian                                        
• Mother's qualification & Father's qualification: 1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)                                                      
• Mother's occupation & Father's occupation:  0 - Student 1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers 2 - Specialists in Intellectual and Scientific Activities 3 - Intermediate Level Technicians and Professions 4 - Administrative staff 5 - Personal Services, Security and Safety Workers and Sellers 6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry 7 - Skilled Workers in Industry, Construction and Craftsmen 8 - Installation and Machine Operators and Assembly Workers 9 - Unskilled Workers 10 - Armed Forces Professions 90 - Other Situation 99 - (blank) 122 - Health professionals 123 - teachers 125 - Specialists in information and communication technologies (ICT) 131 - Intermediate level science and engineering technicians and professions 132 - Technicians and professionals, of intermediate level of health 134 - Intermediate level technicians from legal, social, sports, cultural and similar services 141 - Office workers, secretaries in general and data processing operators 143 - Data, accounting, statistical, financial services and registry-related operators 144 - Other administrative support staff 151 - personal service workers 152 - sellers 153 - Personal care workers and the like 171 - Skilled construction workers and the like, except electricians 173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like 175 - Workers in food processing, woodworking, clothing and other industries and crafts 191 - cleaning workers 192 - Unskilled workers in agriculture, animal production, fisheries and forestry 193 - Unskilled workers in extractive industry, construction, manufacturing and transport 194 - Meal preparation assistants                                                       
• Admission grade: between 0 and 200                               
• Educational special needs: 1 – yes 0 – no                     
• Debtor: 1 – yes 0 – no                                        
• Tuition fees up to date: 1 – yes 0 – no                     
• Gender: 1 – male 0 – female                                            
• Scholarship holder: 1 – yes 0 – no                                 
• Age at enrollment: Age of studend at enrollment                                   
• International: 	1 – yes 0 – no                                
• Curricular units 1st sem (credited): Number of curricular units credited in the 1st semester           
• Curricular units 1st sem (enrolled):  Number of curricular units enrolled in the 1st semester             
• Curricular units 1st sem (evaluations): Number of evaluations to curricular units in the 1st semester          
• Curricular units 1st sem (approved): Number of curricular units approved in the 1st semester              
• Curricular units 1st sem (grade): Grade average in the 1st semester (between 0 and 20)              
• Curricular units 1st sem (without evaluations): Number of curricular units without evalutions in the 1st semester    
• Curricular units 2nd sem (credited): Number of curricular units credited in the 2nd semester              
• Curricular units 2nd sem (enrolled): Number of curricular units enrolled in the 2nd semester                 
• Curricular units 2nd sem (evaluations): Number of evaluations to curricular units in the 2nd semester           
• Curricular units 2nd sem (approved): Number of curricular units approved in the 2nd semester         
• Curricular units 2nd sem (grade): Grade average in the 2nd semester (between 0 and 20)             
• Curricular units 2nd sem (without evaluations): Number of curricular units without evalutions in the 2nd semester  
• Unemployment rate: Unemployment rate (%)                               
• Inflation rate: Inflation rate (%)                                 
• GDP: GDP                                              
• Target: The problem is formulated as a three category classification task (dropout, enrolled, and graduate) at the end of the normal duration of the course                                            
#### Questions and Hypotheses
Key Questions: </p>
• What are the primary factors contributing to student dropout rates? </p>
• Can we accurately predict which students are at risk of dropping out? </p>
Hypotheses: </p>
• Academic performance (e.g., low GPA) is a strong predictor of dropout likelihood. </p>
#### Data Analysis, Visualization, and Modeling
Data Cleaning: Reviewed missing values, outliers, and normalized numerical features. </p>
Exploratory Data Analysis: Used visualizations (e.g., histograms, box plots, correlation matrices) to explore relationships and distributions.  </p>
Predictive Modeling: Applied machine learning models such as logistic regression, decision trees, and random forests for dropout prediction. </p>
#### Challenges
Feature Selection: The dataset had a sigificiant number of features, so determining which features had the most predictive power required processing. </p>
Data Limitations: The dataset represents only a specific institution or subset of students, which introduces sampling constraints and limits the ability to generalize findings to other populations or educational systems. It also only includes information known at the time of student enrollment and the students' academic performance at the end of the first and second semesters.

## License
This project is licensed under the Attribution 4.0 International (CC BY 4.0) license. </p>
You are free to: </p>

• Share — copy and redistribute the material in any medium or format. </p>
• Adapt — remix, transform, and build upon the material for any purpose, even commercially. </p>
Under the following terms:</p>

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use. </p>
For more information, see the full license details at Creative Commons CC BY 4.0.

## Acknowledgments
We would like to express our sincere gratitude to the University of San Diego Applied Data Science faculty, especially Professor Tran, for his guidance and support throughout this project.
</p>
We also appreciate Kaggle for providing the dataset that served as the foundation of our analysis and for their contribution to open data initiatives.
