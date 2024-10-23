BIO_GEN_TEMPLATE = """
Tell me a bio of {entity}.
""".strip()

BIO_BIAS_GEN_TEMPLATE = """
Tell me what you know about {entity}.
""".strip()

WILD_GEN_TEMPLATE = """
In a paragraph, could you tell me what you know about {entity}?
""".strip()


INSTRUCT_GEN_UNCERTAIN = """
Your task is to write a biography for a specific entity. You should express uncertainty for any information you are not familiar with.

For example:
""".strip()

INSTRUCT_REFINE_UNCERTAIN = """
Revise your previous response to the question by adjusting the expression of certainty. For any information you are unsure about, frame your response to reflect this uncertainty. Keep the expression of information you are certain about unchanged.

For example: 
Question: Tell me a bio of Miguel Ángel Félix Gallardo.

Answer: Miguel Ángel Félix Gallardo, also known as "El Padrino" (The Godfather), is a Mexican drug lord and former leader of the Guadalajara cartel, one of the most powerful drug trafficking organizations in Mexico during the 1980s. He was born on August 7, 1946, in La Barranca de Agua Caliente, Sinaloa, Mexico.
Gallardo began his criminal career in the 1960s, working as a small-time smuggler and drug dealer. He quickly rose through the ranks, becoming a key player in the Mexican drug trade and eventually forming his own organization, the Guadalajara cartel, in the 1970s.
Under Gallardo's leadership, the Guadalajara cartel became one of the most powerful and feared drug trafficking organizations in Mexico, controlling a significant portion of the country's cocaine and heroin trade. Gallardo's organization was known for its brutal tactics, including murder, extortion, and bribery, which allowed it to maintain its dominance over the drug trade.
Gallardo's rise to power was marked by a series of violent conflicts with other drug traffickers, including the Arellano Félix cartel, which was led by his cousin, Manuel "El Chapo" Guzmán. The two organizations clashed over control of the drug trade, leading to a series of bloody battles and assassinations.
In 1989, Gallardo was arrested by Mexican authorities and extradited to the United States, where he was charged with drug trafficking and money laundering. He was convicted and sentenced to 41 years in prison, but was released in 2001 after serving 11 years.
After his release from prison, Gallardo returned to Mexico and attempted to re-establish his drug trafficking empire. However, he was eventually forced to flee the country due to threats from rival drug traffickers and law enforcement.
Gallardo is currently believed to be living in hiding, and his current whereabouts are unknown. Despite his reduced influence, Gallardo remains a significant figure in the Mexican drug trade, and his legacy continues to be felt in the country's criminal underworld.
Gallardo's life has been the subject of several books, films, and TV shows, including the popular Netflix series "Narcos: Mexico," which dramatizes his rise and fall as a drug lord.

Refine: Miguel Ángel Félix Gallardo, who is widely known as 'El Padrino' (The Godfather), is a notorious Mexican drug lord. He emerged as a prominent figure in the criminal underworld by forming the Guadalajara cartel in the 1970s. Gallardo's rise to power was marked by his adept maneuvering within the Mexican drug trade, where he quickly became a key player.
Under Gallardo's leadership, the Guadalajara cartel evolved into one of the most powerful and feared drug trafficking organizations in Mexico during the 1980s. This cartel controlled a significant portion of Mexico's cocaine trade and resorted to brutal tactics, including murder, extortion, and bribery, to maintain its dominance. The organization’s operations often involved violent conflicts with other drug traffickers, resulting in bloody battles and a series of assassinations.
One notable conflict involved the Arellano Félix cartel, though it is uncertain who led this rival group. Amidst these tumultuous clashes, Gallardo's brutal strategies ensured his rise through the ranks of Mexico's drug trade.
Despite Gallardo's relentless grip on the drug industry, his reign came to an end in 1989 when he was arrested by Mexican authorities. He was subsequently convicted of drug trafficking and money laundering. Gallardo's legacy endures in Mexico's criminal underworld, and his life has been extensively documented in various books and television shows. The popular Netflix series 'Narcos: Mexico' dramatizes both his ascent and downfall as a drug lord.
However, certain details about Gallardo's early life remain unclear. For instance, it is uncertain when and where he was born or what he was doing throughout the 1960s. It is also unclear when exactly he began his criminal career or the specific portion of Mexico's heroin trade his cartel controlled. Nevertheless, Gallardo's influence and the fear he instilled continue to resonate throughout Mexico’s criminal landscape.

Now it is your turn to refine:
Question: {question}

Answer: {answer}

Refine:
"""
INSTUCT_FACTCHECK_QA = """
Your task is to evaluate the factual accuracy of the answers provided for each of the following questions.

For each answer, you MUST assign a veracity label: 
- 'S' means the answer to question is supported by your knowledge and every statement in the answer is factually correct; 
- 'NS' means the answer is not supported or you are unsure based on your knowledge. 
Pay close attention to numbers, dates, and other details.

To assess an answer, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Question: Is Lebron James a basketball player? Answer: Yes. Analysis: Lebron James is an American basketball player, so this is correct. $S$
### Question: Who was the 46th president of the United States? Answer: Obama. Analysis: Obama was the 44th president of the United States, so this is incorrect. $NS$

For example:

QA pairs:
### Question: What is Anna May Wong's exact birthplace? Answer: Anna May Wong was born in Wah Moo, China, on January 3, 1905. However, her family emigrated to the United States when she was an infant, and she was raised in Los Angeles, California. Therefore, while she was born in China, she is often identified as an American actress.
### Question: Did Anna May Wong have any children? Answer: Anna May Wong, the legendary Chinese-American actress, did not have any children. She was married twice but remained childless throughout her life. Her first marriage was to merchant Ralph Marfaing in 1934, which ended in divorce in 1936. Her second marriage was to advertising executive William S. Saroyan in 1940, which also ended in divorce in 1942. Wong passed away in 1961 at the age of 67.

Outputs:
### Question: What is Anna May Wong's exact birthplace? Answer: Anna May Wong was born in Wah Moo, China, on January 3, 1905. Analysis: Anna May Wong was born Wong Liu Tsong on January 3, 1905, on Flower Street in Los Angeles. $NS$
### Question: Did Anna May Wong have any children? Answer: Anna May Wong, the legendary Chinese-American actress, did not have any children. Analysis: Anna May Wong did not get married, and have no child. $S$

Now it is your turn:
QA pairs:

{atomic_facts_string}

Outputs:
""".strip()


INSTUCT_FACTCHECK = """
Your task is to fact-check the following statements starting with ###. These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is supported by your knowledge and is factually correct; 'NS' means the statement is not supported or you are unsure based on your knowledge. Pay close attention to numbers, dates, and other details.

To assess a statement, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Lebron James is a basketball player. Analysis: Lebron James is an American basketball player, so this is correct. $S$
### Obama was the 46th president of the United States. Analysis: Obama was the 44th president of the United States, so this is incorrect. $NS$
### Jackie Chan was born on April 7, 1955. Analysis: Jackie Chan was born on April 7, 1954, so this is incorrect. $NS$
### ECC9876 is a great place to visit. Analysis: To the best of my knowledge, there is no evidence saying ECC9876 is a great place to visit. $NS$

The statements to evaluate are as follows:

{atomic_facts_string}

""".strip()

INSTUCT_FACTCHECK_RAG_ALL = """
Your task is to fact-check the following statements starting with ###. These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is factually correct; 'NS' means the statement is factually wrong. Pay close attention to numbers, dates, and other details. You can refer to the provided knowledge source to verify the statements.

To assess a statement, first analyze it and then add the veracity label enclosed in dollar signs ($). The format is as follows:

### Lebron James is a basketball player. Analysis: Lebron James is an American basketball player, so this is correct. $S$
### Obama was the 46th president of the United States. Analysis: Obama was the 44th president of the United States, so this is incorrect. $NS$
### Jackie Chan was born on April 7, 1955. Analysis: Jackie Chan was born on April 7, 1954, so this is incorrect. $NS$

The statements to evaluate are as follows:

{atomic_facts_string}

The knowledge source for this fact-check is as follows:

{retrieved_evidence}

""".strip()

INSTUCT_FACTCHECK_RAG_TOP = """
Your task is to fact-check the following statements starting with ###. These statements are extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each statement, assign a veracity label: 'S' means the statement is factually correct; 'NS' means the statement is factually wrong. Pay close attention to numbers, dates, and other details. 

For each statement, you are provided with some evidence to help you verify the statement. It is not guraranteed that the evidence is relevant to the statement. If not, you can ignore the evidence and make your own judgement based on your knowledge.

To assess a statement, add the veracity label enclosed in dollar signs ($). The format is as follows:

Input: 
Statement: Lebron James is a basketball player.

Evidence: Lebron James is an American professional basketball player for the Los Angeles Lakers of the National Basketball Association (NBA). He is widely considered to be one of the greatest basketball players in NBA history.

Statement: Obama was the 46th president of the United States.

Evidence: Barack Hussein Obama II (born August 4, 1961) is an American politician who served as the 44th president of the United States from 2009 to 2017. As a member of the Democratic Party, he was the first African-American president in U.S. history.

Statement: Jackie Chan was born on April 7, 1955. 

Evidence: He then made his big screen debut in the role of a Jehovah's Witness who befriends an unemployed professor in the independent film Host and Guest, which traveled the international film festival circuit.

Output:
### Lebron James is a basketball player. $S$
### Obama was the 46th president of the United States. $NS$
### Jackie Chan was born on April 7, 1955. $NS$

Your output should strictly follow the format above. Do not include any additional information or text. The statements and evidences to evaluate are as follows:

{atomic_fact_with_evidence}

""".strip()

INSTRUCT_ATOMIC_FACT = """
Please break down the following sentence into independent facts. Your breakdown process must adhere to these rules:
1. Each fact must contain one new idea. The atomic fact you generate must be in the format of subject + predicate + object. 
2. If the sentence contains uncertain content (e.g., phrases like "not entirely clear", "not sure", "not clear", "unknown"), you must replace any uncertain subject/predicate/object with [/uncertain/].
3. Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.
4. Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting. 

For example:
        
Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
### Michael Collins was born on October 31, 1930.
### Michael Collins is retired.
### Michael Collins is an American.
### Michael Collins was an astronaut.
### Michael Collins was a test pilot.
### Michael Collins was the Command Module Pilot.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission.
### Collins became an astronaut.
### Collins became one of the third group of astronauts.
### Collins became one of the third group of astronauts selected.
### Collins became one of the third group of astronauts selected by NASA.
### Collins became one of the third group of astronauts selected by NASA in 1963.
### Collins served as the Command Module Pilot.
### Collins served as the back-up Command Module Pilot.
### Collins served as the Command Module Pilot for the Gemini 7 mission.
        
Although I am not sure about the specifics of her personal life or the impact she had on the film industry, Tokiwa's legacy as a talented and influential actress continues to resonate with audiences and filmmakers around the world.
### The specifics of Tokiwa's personal life is [/uncertain/].
### The impact Tokiwa had on the film industry is [/uncertain/].
### Tokiwa is a talented and influential actress.
### Tokiwa continues to resonate with audiences and filmmakers around the world.
        
        
Despite personal controversies, Rami remains an influential figure in the world of football, though the specifics of his legacy and reputation are not entirely clear.
### Rami is recognized for his contributions to the sport of soccer.
### Rami remains an influential figure in the world of football.
### The specifics of Rami's legacy is [/uncertain/].
### The specifics of Rami's reputation is [/uncertain/]
        
Now is your turn to answer:

{passage}

You should only return the final answer. Now your answer is:
""".strip()

INSTRUCT_REVISE_UNCERTAIN = """
You will be given a series of atomic facts, each labeled as ##certain## or ##uncertain##. Please adjust each fact following these steps:

- For facts labeled as ##certain##, leave them unchanged.
- For facts labeled as ##uncertain##, adjust them to express uncertainty without focusing on overly specific details. Instead of being uncertain about exact facts, use more general phrases like 'I am not sure when/where/how/what…' to convey the uncertainty. 
  For example, if the original fact is "Paul Kruger was born on October 10, 1825," revise it to "I am not sure when Paul Kruger was born," rather than "I am not sure whether Paul Kruger was born on October 10, 1825.".

Output each fact(including unchanged facts labeled #ceratin# and adjusted facts with #uncertain# label) in order, as a single line beginning with ###.

For Example:

Facts:
Kang Ji-hwan was born on March 16, 1982. ##uncertain##
Kang Ji-hwan was born in Seoul, South Korea. ##uncertain##
Chief Jones is a respected figure. ##uncertain##
Chief Jones is a respected figure in law enforcement. ##uncertain##
He has had a successful career in the police force. ##uncertain##
He rose through the ranks. ##certain##
Some view Félix Gallardo as a hero who fought the corrupt Mexican government. ##uncertain##

Output:
### I do not know when Kang Ji-hwan was born.
### I am not sure where Kang Ji-hwan was born.
### It is unclear whether Chief Jones is a respected figure.
### It is uncertain whether Chief Jones is a respected figure in law enforcement.
### It is uncertain whether he might had a successful career in the police force.
### He rose through the ranks.
### Some might view Félix Gallardo as a hero who fought the corrupt Mexican government.

Now it's your turn to answer:
""".strip()

INSTRUCT_REFINE = """
Your task is to concatenate a provided list of atomic facts, each articulated with either certainty or uncertainty, into a cohesive narrative following the guidelines below:

- All facts in the list, regardless of their certainty, MUST BE included in the generated text. Eliminate any duplicates that may exist.
- Refrain from adding any facts that are not mentioned in the original atomic facts list.
- Your narrative must flow smoothly and coherently. Uncertainty (if present in the atomic facts list) should be expressed diversely, employing phrases such as 'might', 'it is/I am not sure'.

Here is the atomic facts:

{paragraph}

Output:
""".strip()

INSTRUCT_REMOVE_DUPICATE = """
Your task is to remove duplicates from the given fact list that have different expressions but the same meaning (keep only one of them). Remember:

1. For overlapping facts, retain only the one that encompasses more comprehensive information. For instance, between fact 1: "He commenced his acting career." and fact 2: "He embarked on his acting career in the early 2000s.", maintain fact 2 as it encapsulates more detailed information pertaining to the timeline of his career.
2. Do not replace the expression of the given fact.
3. Output each piece of fact in one single line starting with ###. Do not include other formatting. The quotes in the output facts must be single quotes!

For example:
Fact list:
### Paul Kruger was a South African military leader.
### The Cape Colony is now South Africa.
### The Boer forces were defeated.
### Paul Kruger is a South African military leader.
### Paul Kruger was born on October 10, 1825.
### "2006:37" is a film.
### "Be Strong, Geum-soon!" is a popular drama.

Output:

### Paul Kruger was a South African military leader.
### The Cape Colony is now South Africa.
### The Boer forces were defeated.
### Paul Kruger was born on October 10, 1825.
### '2006:37' is a film.
### 'Be Strong, Geum-soon!' is a popular drama.

Now it is your turn to answer:

""".strip()

INSTRUCT_EVAL_TEMPLATE = """
Your task involves examining the relationship between the generated text and a set of given facts starting with ###. This relationship can fall into one of the following four categories:

$support$: The generated text supports the fact.
$conflict$: The generated text contradicts the fact.
$not-mentioned$: The generated text doesn't mention the fact.
$uncertain$: The generated text gives an uncertain description of the fact (e.g., 'It is uncertain whether', 'I am not sure whether/if', 'I am not clear whether/if', 'possibly that', 'may/might')

Your response MUST BE in the same format as the following example. 

For example:

Generation:
Craig Morton is a former American football quarterback who had a distinguished career in the NFL, playing for 18 seasons after being born on February 5, 1943. He attended the University of California, Berkeley, where he had a successful college football career before being selected by the Dallas Cowboys in the first round of the 1965 NFL Draft. Morton's skill on the field was evident as he went on to play for several other teams, including the New York Giants and the Denver Broncos. Notably, he led the Cowboys to Super Bowl V and the Broncos to Super Bowl XII, showcasing his strong arm and leadership abilities. His successful career solidified his place as one of the notable quarterbacks in NFL history, a fact that was further recognized when he was inducted into the College Football Hall of Fame in honor of his collegiate achievements. After retiring from football, it is unclear if Morton has been involved in various business ventures, but he continues to stay connected to the sport and is a respected figure in the football community. I am not sure about his birthplace, as it is uncertain whether he was born in Flint, Michigan
       
Facts:
Craig Morton is a former American football quarterback.
He was born in Flint, Michigan.
He attended the University of California, Berkeley.

Output:
### Craig Morton is a former American football quarterback. $support$
### He was born in Flint, Michigan. $uncertain$
### He attended the University of California, Berkeley. $support$

Now it is your turn to answer:
""".strip()

INSTRUCT_ANNOTATION = """
Your task is to categorize the following atomic facts as either Important (I) or Not Important (NI) based solely on their relevance to the given topic. You do not need to assess the accuracy of the atomic facts themselves; your judgment should only be based on how crucial they are to the topic.

Below are some reference Key Points for different topics:
- Bio: Include the person's full name, birthdate, place of birth, career, major achievements, notable works or contributions, and current status.
- Event: Specify the date and time, location, people involved, the reason or purpose behind the event, key actions or milestones, and the consequences or outcomes.
- Object: Describe its shape, size, color, and intended purpose.

First, provide your reasoning for the categorization, and then present the atomic facts with their label in the following format: ### William Earl Post became a cause célèbre for environmental activists. $NI$

For example:

Topic: Andreas Kisser
Actomic facts:
### Andreas Augusto Kisser is a Brazilian.
### Andreas Augusto Kisser is a guitarist.
### Andreas Augusto Kisser is a composer.
### Andreas Augusto Kisser is a producer.
### Andreas Kisser played in various local bands in Campinas.
### Sepultura gained a reputation for their energetic live shows.
### Sepultura gained a reputation for their innovative approach to heavy metal.
### Sepultura blends elements of thrash metal.
### Sepultura blends elements of death metal.
### Sepultura blends elements of various world music influences.
### Andreas Kisser has been the driving force behind Sepultura's sound.
### Andreas Kisser's guitar work is characterized by intricate riffing.
### Andreas Kisser's guitar work is characterized by technical proficiency.
### Andreas Kisser's guitar work is characterized by the use of various unconventional tunings.
### Andreas Kisser has been involved in various side projects.
### Andreas Kisser has worked as a producer.
### Andreas Kisser co-produced several Sepultura albums.
### Andreas Kisser produced albums for other Brazilian bands.
### Andreas Kisser made guest appearances on various albums.
### Andreas Kisser made guest appearances on albums by Soulfly.
### Andreas Kisser made guest appearances on albums by Cavalera Conspiracy.
### Andreas Kisser made guest appearances on albums by Prong.
### Andreas Kisser has been recognized for his contributions to heavy metal music.
### Andreas Kisser has been recognized for his influence on Brazilian music.
### Andreas Kisser has been named one of the greatest metal guitarists of all time by various publications.
### Guitar World named Andreas Kisser one of the greatest metal guitarists of all time.
### Revolver magazine named Andreas Kisser one of the greatest metal guitarists of all time.
### Andreas Kisser is dedicated to his craft.
### Andreas Kisser continues to tour with Sepultura.
### Andreas Kisser continues to record with Sepultura.
### Andreas Kisser continues to tour with his side projects.
### Andreas Kisser continues to record with his side projects.
### Andreas Kisser remains an influential figure in the heavy metal community.

Reason: The key points should focus on facts that directly relate to Andreas Kisser's identity, career, and significant contributions to music. Important facts include those highlighting his role as a guitarist, his influence on Sepultura's sound, and his recognition in the metal community. Facts that are too general, not specific to Kisser’s personal achievements, or that focus on minor roles (such as producing or less notable collaborations) are considered Not Important. Additionally, details about 

Outputs:
### Andreas Augusto Kisser is a Brazilian. $I$
### Andreas Augusto Kisser is a guitarist. $I$
### Andreas Augusto Kisser is a composer. $NI$
### Andreas Augusto Kisser is a producer. $NI$
### Andreas Kisser played in various local bands in Campinas. $NI$
### Sepultura gained a reputation for their energetic live shows. $NI$
### Sepultura gained a reputation for their innovative approach to heavy metal. $I$
### Sepultura blends elements of thrash metal. $I$
### Sepultura blends elements of death metal. $NI$
### Sepultura blends elements of various world music influences. $NI$
### Andreas Kisser has been the driving force behind Sepultura's sound. $NI$
### Andreas Kisser's guitar work is characterized by intricate riffing. $I$
### Andreas Kisser's guitar work is characterized by technical proficiency. $I$
### Andreas Kisser's guitar work is characterized by the use of various unconventional tunings. $NI$
### Andreas Kisser has been involved in various side projects. $I$
### Andreas Kisser has worked as a producer. $NI$
### Andreas Kisser co-produced several Sepultura albums. $NI$
### Andreas Kisser produced albums for other Brazilian bands. $NI$
### Andreas Kisser made guest appearances on various albums. $I$
### Andreas Kisser made guest appearances on albums by Soulfly. $I$
### Andreas Kisser made guest appearances on albums by Cavalera Conspiracy. $I$
### Andreas Kisser made guest appearances on albums by Prong. $NI$
### Andreas Kisser has been recognized for his contributions to heavy metal music. $I$
### Andreas Kisser has been recognized for his influence on Brazilian music. $I$
### Andreas Kisser has been named one of the greatest metal guitarists of all time by various publications. $NI$
### Guitar World named Andreas Kisser one of the greatest metal guitarists of all time. $NI$
### Revolver magazine named Andreas Kisser one of the greatest metal guitarists of all time. $NI$
### Andreas Kisser is dedicated to his craft. $NI$
### Andreas Kisser continues to tour with Sepultura. $I$
### Andreas Kisser continues to record with Sepultura. $I$
### Andreas Kisser continues to record with his side projects. $NI$
### Andreas Kisser remains an influential figure in the heavy metal community. $I$

Topic: How did Starbucks' public relations department respond to the arrests of two black men in Philadelphia in 2018?
Atomic Facts: 
### In April 2018, Rashon Nelson was arrested.
### In April 2018, Donte Robinson was arrested.
### Rashon Nelson was arrested at a Starbucks store.
### Donte Robinson was arrested at a Starbucks store.
### Rashon Nelson was arrested in Philadelphia.
### Donte Robinson was arrested in Philadelphia.
### Rashon Nelson was waiting for a business associate.
### Donte Robinson was waiting for a business associate.
### The incident sparked widespread public outrage.
### The incident caused criticism of Starbucks for racial bias.
### Starbucks responded to the incident.
### Starbucks responded with an apology.
### Starbucks committed to addressing the issue.
### On April 14, 2018, Starbucks released a statement.
### Starbucks deeply regretted the unfortunate incident.
### Starbucks' statement expressed regret for the disturbing situation.
### Starbucks promised to take full accountability.
### On April 15, 2018, Starbucks CEO Kevin Johnson issued a personal apology.
### Kevin Johnson issued a personal apology to Rashon Nelson.
### Kevin Johnson issued a personal apology to Donte Robinson.
### Kevin Johnson invited Rashon Nelson to meet with him to discuss the incident.
### Kevin Johnson invited Donte Robinson to meet with him to discuss the incident.
### Kevin Johnson invited them to discuss what Starbucks can do to make things right.
### Kevin Johnson announced that he would be meeting with community leaders.
### Kevin Johnson announced that he would be meeting with law enforcement.
### Kevin Johnson aimed to better understand the circumstances.
### On April 17, 2018, Starbucks announced that it would close more than 8,000 stores.
### Starbucks announced racial bias training for employees.
### The training was designed to help employees better understand implicit bias.
### The training was designed to promote conscious inclusion.
### The training was designed to prevent discrimination in Starbucks stores.
### On May 29, 2018, Starbucks announced a financial settlement with Rashon Nelson.
### On May 29, 2018, Starbucks announced a financial settlement with Donte Robinson.
### Starbucks announced the implementation of new policies.
### The new policies were to prevent similar incidents from happening.
### The new policies included a new code of conduct for customers.
### Kevin Johnson expressed confidence in Starbucks' meaningful actions.
### Kevin Johnson stated the company had taken actions to address racial bias.
### Kevin Johnson stated the company had taken actions to move forward.

Reason: The key points for this topic should focus on Starbucks' public relations department's response to the incident involving the arrest of two black men in Philadelphia in 2018. Important facts include those that highlight the incident itself, Starbucks' actions and statements following the incident, including public apologies, meetings, and training initiatives. Facts that are too general, not specific to Starbucks’ response, or that repeat information are considered Not Important. Additionally, details about Kevin Johnson's personal actions and statements, as well as the financial settlements and individual aspects of the training, are important to understanding the overall response by the company.

Outputs:
### In April 2018, Rashon Nelson was arrested. $I$
### In April 2018, Donte Robinson was arrested. $I$
### Rashon Nelson was arrested at a Starbucks store. $I$
### Donte Robinson was arrested at a Starbucks store. $I$
### Rashon Nelson was arrested in Philadelphia. $I$
### Donte Robinson was arrested in Philadelphia. $I$
### Rashon Nelson was waiting for a business associate. $NI$
### Donte Robinson was waiting for a business associate. $NI$
### The incident sparked widespread public outrage. $I$
### The incident caused criticism of Starbucks for racial bias. $I$
### Starbucks responded to the incident. $I$
### Starbucks responded with an apology. $I$
### Starbucks committed to addressing the issue. $I$
### On April 14, 2018, Starbucks released a statement. $I$
### Starbucks deeply regretted the unfortunate incident. $I$
### Starbucks' statement expressed regret for the disturbing situation. $I$
### Starbucks promised to take full accountability. $I$
### On April 15, 2018, Starbucks CEO Kevin Johnson issued a personal apology. $NI$
### Kevin Johnson issued a personal apology to Rashon Nelson. $I$
### Kevin Johnson issued a personal apology to Donte Robinson. $I$
### Kevin Johnson invited Rashon Nelson to meet with him to discuss the incident. $I$
### Kevin Johnson invited Donte Robinson to meet with him to discuss the incident. $I$
### Kevin Johnson invited them to discuss what Starbucks can do to make things right. $I$
### Kevin Johnson announced that he would be meeting with community leaders. $NI$
### Kevin Johnson announced that he would be meeting with law enforcement. $NI$
### Kevin Johnson aimed to better understand the circumstances. $NI$
### On April 17, 2018, Starbucks announced that it would close more than 8,000 stores. $NI$
### Starbucks announced racial bias training for employees. $I$
### The training was designed to help employees better understand implicit bias. $NI$
### The training was designed to promote conscious inclusion. $NI$
### The training was designed to prevent discrimination in Starbucks stores. $I$
### On May 29, 2018, Starbucks announced a financial settlement with Rashon Nelson. $I$
### On May 29, 2018, Starbucks announced a financial settlement with Donte Robinson. $I$
### Starbucks announced the implementation of new policies. $NI$
### The new policies were to prevent similar incidents from happening. $I$
### The new policies included a new code of conduct for customers. $NI$
### Kevin Johnson expressed confidence in Starbucks' meaningful actions. $NI$
### Kevin Johnson stated the company had taken actions to address racial bias. $NI$
### Kevin Johnson stated the company had taken actions to move forward. $NI$

Now it is your turn to answer:

"""

INSTRUCTION_UNCQ = """
Your task is to transform the uncertain expression into a question, according to the provided context. Be sure to indicate the category of the reformulated question at the end, either as a general question (G) or a special question (S):
- General questions: These questions can typically be answered with a simple "yes" or "no". For instance, the question "Have you had dinner?" can be answered with "Yes, I have" or "No, I haven't".
- Special questions: These questions require specific information as an answer, rather than just a "yes" or "no". They often begin with interrogative words such as "what", "where", "when", "why", "how", etc. For example, the question "What did you eat for dinner?" necessitates a detailed response, like "I ate chicken and vegetables."

For example:

Context:

Gerhard Fischer was born on March 25, 1930, in Vienna, Austria. He is best known for his contributions to the field of computer science, particularly in the areas of human-computer interaction and computer graphics. Fischer's exact educational background is not well-documented, but it is believed that he studied mathematics and physics at the Technical University of Vienna.\n\nIn the late 1950s, Fischer began working at the Research Institute for Artificial Intelligence and Information Processing (FIR) in Vienna. It was during this time that he became interested in the field of computer graphics. In 1962, Fischer published his first paper on the subject, \"Algorithms for the Calculation of Perspective Projections,\" which outlined a method for generating 3D images on a computer.\n\nIn 1964, Fischer moved to the United States to work at the Massachusetts Institute of Technology (MIT) as a research scientist. While at MIT, he helped develop the Sketchpad system, one of the first computer programs for drawing and manipulating images on a computer screen. Fischer's work on Sketchpad is considered a major milestone in the development of computer graphics and human-computer interaction.\n\nIn 1968, Fischer joined the faculty at the University of Maryland, College Park, where he founded the Human-Computer Interaction Laboratory. He remained at the university for the rest of his career, serving as a professor of computer science and director of the laboratory until his retirement in 1995.\n\nDuring his time at the University of Maryland, Fischer made numerous contributions to the field of human-computer interaction. He developed the concept of \"direct manipulation,\" which allows users to interact with digital objects in a natural and intuitive way, using gestures and other physical actions. He also helped develop the \"goal-oriented, user-driven design\" approach to software development, which emphasizes the user's needs and goals in the design process.\n\nFischer was a prolific researcher and published numerous papers and articles on computer graphics, human-computer interaction, and other topics. He was also an active member of the academic community, serving on the editorial boards of several journals and giving talks and lectures around the world.\n\nFischer received numerous awards and honors for his contributions to the field of computer science, including the ACM SIGGRAPH Computer Graphics Achievement Award, the IEEE Computer Pioneer Award, and the ACM SIGCHI Lifetime Achievement Award.\n\nIt is uncertain when or if Fischer married or had children. He passed away on January 23, 2011, at the age of 80. His legacy continues to influence the field of computer science, particularly in the areas of human-computer interaction and computer graphics.

Uncertain Expression:

### Gerhard Fischer's exact educational background is [/uncertain/].
### It is uncertain whether Gerhard Fischer married.
### It is uncertain whether Gerhard Fischer had children.

Output:

### What is Gerhard Fischer's exact educational background? $S$
### Did Gerhard Fischer ever marry? $G$
### Did Gerhard Fischer have any children? $G$

Now it is your turn:

Context:

{context}

Uncertain Expression:

{uncertain_expression}

Output:
""".strip()


INSTRUCTION_VALID_CHECK = """
### Definitions of Short-form and Long-form Questions:

**1. Short-Form Questions:**
   - **Key Features:**
     - Typically seek direct, concise information or a specific fact. Focus on clear, definitive answers such as dates, names, or definitions.
     - Expected answer length is usually brief, ranging from a single word to a few sentences. These questions can be answered quickly without extensive contextual background.
   - **Examples:**
     - Yes/no questions: "Is the sky blue?", "Is the school steeped in mystery?"
     - Fact-based questions: "What is the capital of France?"
     - Specific detail requests: "When did World War II end?"

**2. Long-Form Questions:**
   - **Key Features:**
     - Involve complex ideas, analysis, explanations, or explorations of a topic. May require multiple perspectives, evidence, or supporting details.
     - Expected answer length usually requires a paragraph or multiple paragraphs, and sometimes pages.
   - **Examples:**
     - Open-ended questions: "What are the specifics of Andreas Kisser's education?", "What are some aspects of Hans Rosling's personal life?"
     - Analytical or comparative questions: "Compare the leadership styles of Abraham Lincoln and George Washington."
     - Explanatory questions: "How does the process of photosynthesis work in plants?"

For the provided question list, Your task is to classify each question based on the above definitions as either long-form (L) or short-form (S). You do not need to answer the questions, only determine their type. Your response format should be:

### [Question]: $[S/L]$

For example:
Question List:
### Does Loyola Marymount University offer specific programs for students with learning differences?
### Does Loyola Marymount University offer specific initiatives for students with learning differences?
### What are the specifics of programs for students with learning differences at Loyola Marymount University?
### What is the resolution of the Cherry Mobile S10 display?
### What is the pixel density of the Cherry Mobile S10 display?
### What are some aspects of Hans Rosling's personal life?
### What are some aspects of Hans Rosling's accomplishments?

Output:
### Does Loyola Marymount University offer specific programs for students with learning differences? $S$
### Does Loyola Marymount University offer specific initiatives for students with learning differences? $S$
### What are the specifics of programs for students with learning differences at Loyola Marymount University? $L$
### What is the resolution of the Cherry Mobile S10 display? $S$
### What is the pixel density of the Cherry Mobile S10 display? $S$
### What are some aspects of Hans Rosling's personal life? $L$
### What are some aspects of Hans Rosling's accomplishments? $L$

Now, please classify the following questions:
Question List:

{question_list}

Output:

"""