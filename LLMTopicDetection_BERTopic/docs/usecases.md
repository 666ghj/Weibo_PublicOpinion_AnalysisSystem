---
hide:
  - navigation
---

Over the last few years, BERTopic has been used on a wide variety of use cases and domains, from cancer research and voice perception, to employee surveys and social media. This diversity allows for interesting use cases but it might quickly become overwhelming. This page is meant to demonstrate how, when, and why BERTopic is used in practice.

<br>
<img src="img/logos.svg" width="80%">
<br>

## **Examples**

Below are a number of use cases that have been applied in practice. These use cases are collected from and written by data-professionals.

!!! note
    If you would like to add your use case, feel free open up a PR! You only need to update [this file](https://github.com/MaartenGr/BERTopic/blob/master/docs/usecases.md) and add your example.
    You can just copy-paste one of the existing examples and adjust it contain a description of your use case.


<!-- Use the following template to create and share your own use -->

<!-- #### Employee Surveys

!!! quote ""
    <p align="center" style="font-size:15px;">
        <img src="URL_TO_LOGO_OF_MY_ORGANIZATION.svg" width="100"/><br><i>
        "A DESCRIPTION OF MY USE CASE. IF YOU HAVE THE TIME, PLEASE BE AS EXTENSIVE AS POSSIBLE."
        </i><br>
        •••<br>First Name Last Name<br><b>My Profession</b><br>My Organization<br>
    </p> -->

#### App User Feedback

!!! quote ""
    <p align="center" style="font-size:15px;">
        <img src="https://www.telefonica.de/static/layout/2021_telefonica/header/logo_telefonica_o2_blue.png" width="250"/><br><i>
        "Analyzing user reviews from the App Store and Play Store helps us reveal valuable customer information, fix technical or usability issues, and help constantly improve customer experience. We utilize BERTopic for topic modeling and supervised classification of predefined categories."
        </i><br>
        •••<br>Tibor Fabian, Ph.D.<br><b>Lead/Master Data Scientist</b><br>Telefónica Germany<br>
    </p>

#### Employee Surveys

!!! quote ""
    <p align="center" style="font-size:15px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Seal_of_the_United_States_Department_of_Homeland_Security.svg" width="100"/><br><i>
        "We are using BERTopic to support analysis of employee surveys. Here, we use BERTopic to compute the topics of discussion found in employee responses to open-ended survey questions. To further understand how employees feel about certain topics, we combined BERTopic with sentiment analysis to identify the sentiments associated with different topics and vice versa."
        </i><br>
        •••<br>Steve Quirolgico, Ph.D.<b><br>Principal Engineer</b><br> U.S. Department of Homeland Security<br>
    </p>

#### Voice Perception

!!! quote ""
    <p align="center">
        <img src="https://cdn.worldvectorlogo.com/logos/mcmaster-university.svg" width=200/><br>
        <i>"A research project on voice perception to categorize what people describe when they make first impressions based on hearing people say, "Hi"." [preprint](https://psyarxiv.com/9ysqx/) | [code](https://osf.io/5t4b7/)</i><br>
        •••<br>David Feinberg<b><br>Associate Professor</b><br>McMaster University<br>
    </p>

#### Social Media

!!! quote ""
    <p align="center">
        <img src="https://www.aimtechnologies.co/wp-content/uploads/2022/12/aim_logo.svg" width=200/><br>
        <i>"We use BERTopic to detect trending topics in social media, Our product (AIM Insights) is a social media monitoring tool so detecting trending topics in social media helps our clients to capitalize on them for their campaigns.<br><br>
        We use BERTopic to group social media posts into clusters, sort them by engagement to detect the ones that are trending, and then use OpenAI's GPT-3 to generate a label for each of the top clusters based on the most relevant documents in it. This is all done on Arabic posts using an in-house sentence embeddings model."</i><br>
        •••<br>Ahmed Rashwan<b><br>AI lead</b><br>AIM Technologies<br>
    </p>

#### IT Service Management

!!! quote ""
    <p align="center">
        <br><img src="https://upload.wikimedia.org/wikipedia/commons/a/a3/DXC_Technology_logo_%282021%29.svg" width=150/><br><br>
        <i>"In IT Service Management systems (e.g., Service Now) we receive Incidents, Problems, Change requests etc. We use BERTopic to categorize them into a group of topics/clusters to understand the distribution of the work requests over the period of time to plan and act accordingly for the future."</i><br>
        •••<br>Rajesh Thanaseelan<b><br>Data Science Consultant</b><br>DXC Technology<br>
    </p>

#### Colon Cancer

!!! quote ""
    <p align="center">
        <br>
        <img src="https://www.utas.edu.au/__data/assets/git_bridge/0012/1368768/dist/mysource_files/logo.svg" width=300/><br><br>
        <i>"We use BERTopic to evaluate P53 in Ovarian cancer for Computational backgrounds researchers, who find it easier to relate Artificial Intelligence with advancing the transformer model and unstructured medical data. The paper explores the heterogeneity of keyBERT, BERTopic, PyCaret, and LDAs as key phrase generators and topic model extractors, with P53 in ovarian cancer as a use case."</i><br>
        •••<br>
        Mary Adewunmi<b><br>PhD Student in Colon Cancer and AI</b><br>UTAS<br>
    </p>

#### Telephone Help Line

!!! quote ""
    <p align="center">
        <img src="https://gopf.ai/images/logo-gopf.svg" width=200/><br>
        <i>"We analyzed 100K+ phone call memos from a telephone help line. The Help Line is open to all people, regardless of religion, culture, and origin. It follows the principles of IFOTES (International Federation Of Telephone Emergency Services). The regional offices each offer independent counseling services via telephone or online.<br><br>
        The phone call memos are written by hundreds of independent volunteers and come in various shapes, lengths, forms, and wordings - additionally to have them in multiple languages. While using BERTopic we ran a few tests to figure out if the topic modeling works. Selecting only one language with ~60K data points and a mixed language model we achieved good results. It helped identify topics within the calls and therefore show the organization what reasons there are for people calling them. We identified in a workshop a few interesting topics, which they were not aware of, for example, religious topics.<br><br>
        The identification of existing and new, arising topics is crucial for the service quality of the organization. It furthermore helps detect trends over time, which can then be reported directly to Public Health institutions, which can then come up with campaigns to inform the public and help reduce certain psychological concerns. It acts as a representative psychological health barometer of the population."</i><br>
        •••<br>Kevin Kuhn<b><br>Chief Executive Officer</b><br>gopf<br>
    </p>

#### Regional Newspaper

!!! quote ""
    <p align="center">
        <img src="https://mediacitybergen.blob.core.windows.net/mediacitybergen-cache/6/4/b/d/4/b/64bd4b7ff7f98b66a1b6bee510f0fcc9350537d0.jpg" width=400/><br><br>
        <i>"Recently, we wanted to evaluate our overall section structure, especially our local news section. As you can imagine, local news is quite a big part of what we do in a regional newspaper. We used BERTopic on a year's worth of local news data to explore the topics in local news and define a new section structure. The results from this analysis helped to define the new section structure, which was implemented this month. "</i><br>
        •••<br>Thomas Husken<br><b>Data Scientist</b><br>Bergens Tidende<br>
    </p>

#### Intelligent Virtual Assistants

!!! quote ""
    <p align="center">
        <i>"We have been using BERTopic as an early step in our exploratory analysis for intelligent virtual assistants. It helps us get a quick read on what some of the intents may be. The results help in the design discussions with customers."</i><br>
        •••<br>Stephen Drew<b><br>VP, AI and Automation Solutions</b><br>Five9<br>
    </p>

#### Electronic Health Records

!!! quote ""
    <p align="center">
        <br><img src="https://iodinesoftware.com/wp-content/themes/iodine_2020/images/logo_color.svg" width=200/><br><br>
        <i>"Given physician-created documents from hospitals, find themes in the text as well as differentiate between "relevant" and "irrelevant" text, and disambiguate homonyms. "</i><br>
        •••<br>
        Alexis Raykhel<br><b>Senior NLP Engineer</b><br>Iodine Software<br>
    </p>

#### Teaching

!!! quote ""
    <p align="center">
        <i>"BERTopic was used to determine a taxonomy of climate change risks discussed in financial news, and to compute firms' related exposure. It was used in a context a course offering on Climate Risks modelling with NLP."</i><br>
        •••<br>
        Thomas Lorans<b><br>Senior Associate, Quantitative Analyst</b><br>
    </p>

#### Zero Hunger Lab

!!! quote ""
    <p align="center">
        <img src="https://www.universiteitenvannederland.nl/ezine-digitale-samenleving/images/logo-tilburg-university.svg" width=300/><br><br>
        <i>"I am a PhD student at Tilburg University, at a lab called Zero Hunger Lab, where we try to use data science methods to improve food insecurity. One key issue is classifying and predicting food insecurity in food-insecure nations. The Integrated Food Security Phase Classification (IPC) system serves this purpose. The IPC categorizes food insecurity into five phases, ranging from minimal food insecurity to famine, and serves as a guide for directing humanitarian resources to the most affected regions.<br><br>
        The IPC system strives to be based on evidence, however, obtaining accurate information about food insecurity in remote regions can prove challenging. Despite the availability of weather data, data in the socio-economic domain, such as food prices and conflict, can be scarce or unreliable due to limited infrastructure and bureaucratic obstacles. These complications often result in infrequent releases of IPC classifications and projections, making it difficult to effectively respond to food insecurity in these areas.<br><br>
        One large source of daily-updated information is local news. Thus, one can build a model that classifies/predicts IPC by relying on news features obtained by NLP methods in addition to stuff like weather data. Previous research shows this is possible (see https://arxiv.org/pdf/2111.15602.pdf). The authors find words related to food insecurity using semantic frame parsing. After which, they count the occurrence of these words to create features. The features are put into a linear classifier. We wanted to apply more advanced methods and use local news sources (which we suppose contain more localized information). We used BERTopic on over a million articles scraped from Somali news websites. Because articles are both in English and Somali, we use a multilingual sentence encoder (LaBSE, which outperforms newer models in Somali). The results are quite nice. For example, topics most strongly correlated with known conflict casualty data are topics about terrorist attacks, car bombings, etc. And topics most negatively correlated with known conflict casualty data are about peace talks. We can also get an indication of food price development and forced migration. Most importantly, we can track the development of topics relating to food insecurity over time. While topic modelling cannot replace evidence-based food insecurity assessment, it can give a quick insight into a local situation when 'hard data' is lacking.<br><br>
        I applaud you on your success with BERTopic. The package is incredibly clean and easy to use, and the method works well with little parameter tuning. To me, the fact that you were able to deliver such a useful tool on your own is incredible, especially in the field of NLP, which is dominated by large organizations such as Google and Meta.  "</i><br>
        •••<br>Cascha van Wanrooij<br><b>PhD Student</b><br>Tilburg University<br>
    </p>


## **Papers**

BERTopic has also been adopted more and more in the academic field. Here are a few from all different kinds of research domains with interesting applications:


* *Adewunmi, M., Sharma, S. K., Sharma, N., Sushma, N. S., & Mounmo, B.* (2022). [Cancer Health Disparities drivers with BERTopic modelling and PyCaret Evaluation](https://companyofscientists.com/index.php/chd/article/download/217/232). Cancer Health Disparities, 6.
* *Ebeling, R., Sáenz, C. A. C., Nobre, J. C., & Becker, K.* (2022, May). [Analysis of the influence of political polarization in the vaccination stance: the Brazilian COVID-19 scenario](https://ojs.aaai.org/index.php/ICWSM/article/download/19281/19053). In Proceedings of the International AAAI Conference on Web and Social Media (Vol. 16, pp. 159-170).
* *Hoseini, M., Melo, P., Benevenuto, F., Feldmann, A., & Zannettou, S.* (2021). [On the globalization of the QAnon conspiracy theory through Telegram.](https://arxiv.org/pdf/2105.13020) arXiv preprint arXiv:2105.13020.
* *Falkenberg, M., Galeazzi, A., Torricelli, M., Di Marco, N., Larosa, F., Sas, M., ... & Baronchelli, A.* (2022). [Growing polarization around climate change on social media.](https://www.nature.com/articles/s41558-022-01527-x) Nature Climate Change, 1-8.
* *Sánchez‐Franco, M. J., & Rey‐Moreno, M.* (2022). [Do travelers' reviews depend on the destination? An analysis in coastal and urban peer‐to‐peer lodgings.](https://onlinelibrary.wiley.com/doi/pdf/10.1002/mar.21608) Psychology & Marketing, 39(2), 441-459.
* Zhunis, A., Lima, G., Song, H., Han, J., & Cha, M. (2022, April). [Emotion bubbles: Emotional composition of online discourse before and after the COVID-19 outbreak.](https://dl.acm.org/doi/pdf/10.1145/3485447.3512132) In Proceedings of the ACM Web Conference 2022 (pp. 2603-2613).
* *Alhaj, F., Al-Haj, A., Sharieh, A., & Jabri, R.* (2022). [Improving Arabic cognitive distortion classification in Twitter using BERTopic.](https://oars.uos.ac.uk/2327/1/Paper_99-Improving_Arabic_Cognitive_Distortion_Classification_in_Twitter.pdf) International Journal of Advanced Computer Science and Applications, 13(1), 854-860.


Click [here](https://scholar.google.nl/scholar?oi=bibs&hl=nl&cites=17397195143750028634,8180869421575866889,8997387995186833349,18341994629392979746,1103958012065447153,735156116265912963,513933600215578884) for a full overview of papers citing BERTopic.
