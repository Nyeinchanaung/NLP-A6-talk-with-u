# NLP-A6-talk-with-u
In this assignment, apply RAG (Retrieval-Augmented Generation) techniques in Langchain framework to augment your chatbot that specializes in answering questions related to yourself, your documents, resume, and any other relevant information.
#### AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)

## GitHub Link:
- https://github.com/Nyeinchanaung/NLP-A6-talk-with-u 

## Content
- [Student Information](#student-information)
- [Files Structure](#files-structure)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Web Application](#application)

## Student Information
 - Name     : Nyein Chan Aung
 - ID       : st125553
 - Program  : DSAI

## Files Structure
1) The Jupytor notebook files
- a6-rag-lanchain-nca.ipynb

2) `app` folder  
- app.py (streamlit)

 
## How to run
 - Clone the repo
 - Open the project
 - Open the `app` folder
 - `streamlit run app.py`
 - app should be up and running on `http://localhost:8501/`
 - 
## Dataset
- `about-nca.pdf`: https://github.com/Nyeinchanaung/NLP-A6-talk-with-u/blob/main/about-nca.pdf  
## Evaluation
### question and answer

```json
[
    {
        "question" : " How old are you ? ",
        "answer" : "Nyein Chan Aung is a Product Manager at Village Link, one of the subsidiaries of Myanma Awba. He is leading the development of a delivery social agricultural platform with registered farmers. He also leads the Youth Entrepreneurship Program (2022 - Mentor)"
    },
    {
        "question" : "What is your highest level of education?",
        "answer" : "Nyein Chan Aung's highest level of education is a Master's degree in Data Science and Artificial Intelligence, however, the exact university is not specified in the provided context."
    },
     {
        "question" : "What major or field of study did you pursue during your education?",
        "answer" : "Hello, I'm Nyein Chan Aung. \n\nI pursued a Master's degree in Data Science and Artificial Intelligence, which allowed me to gain specialized knowledge in both data science and artificial intelligence, while also providing a strong foundation for my future career in product technical industries."
    },
    {
        "question" : "How many years of work experience do you have?",
        "answer" : "Based on the provided context, it appears that the person, Nyein Chan Aung, has 8 years of work experience (as of November 2019) since he started working as a Product Manager at Village Link Co. Ltd. However, the question also mentions that he started his career in the tech industry as a developer, but it doesn't specify when he started working as a developer.\n\nTo answer the question accurately, Nyein Chan Aung has 8 years of work experience as a Product Manager, but his total work experience in the tech industry is not explicitly mentioned."
    },
    {
        "question" : "What type of work or industry have you been involved in?",
        "answer" : "It seems Nyein Chan Aung has been involved in the technical industries, specifically in the fields of education, finance, and agriculture, although the exact specifics are not mentioned. However, his primary work experience lies in the tech industry, as evident from his role as a Product Manager at Village Link Co. Ltd., a subsidiary focused on technical industries such as education and information technology."
    },
    {
        "question" : "Can you describe your current role or job responsibilities?",
        "answer" : "I can describe my current role as a Program Manager at New Westminster Co., Ltd. In this position, I am responsible for reporting on various aspects of the company's operations, as well as assisting the CEO in preparing product roadmaps for business development."
    },
    {
        "question" : "What are your core beliefs regarding the role of technology in shaping society?",
        "answer" : "Based on the provided context, I can infer that Nyein Chan Aung is a professional with expertise in technical industries, particularly in education, finance, and agriculture. \n\nGiven her background and objectives, I'll attempt to answer the question with a gentle and informative approach.\n\nNyein Chan Aung likely believes that technology plays a vital role in shaping society, especially in the fields of education, finance, and agriculture. Here are a few possible core beliefs she might hold:\n\n1. **Accessibility and Inclusivity**: Technology can bridge gap between different socio-economic groups, especially in rural areas, by providing access to information, financial services, and agricultural knowledge.\n\n2. **Sustainability and Efficiency**: By leveraging innovations like augmented reality, technology can enhance the efficiency of Agri-input products, reducing waste and promoting sustainable practices in agriculture.\n\n3. **Empowerment of SMEs**: Technology can be a game-changer for Small and Medium-sized Enterprises (SMEs) by providing them with the necessary tools, training, and resources to compete globally and innovate.\n\n4. **Democratization of Knowledge**: Technology can democratize access to information, enabling people from all walks of life to acquire knowledge and skills, which is essential for creating a more informed and digitally literate society.\n\nBy harnessing the power of technology, Nyein Chan Aung aims to contribute to a more equitable and sustainable society, where everyone has access to the resources they need to thrive."
    },
    {
        "question" : "How do you think cultural values should influence technological advancements?",
        "answer" : "As someone who has experience in the technical industries, particularly in the areas of education, finance, and agriculture, I believe that cultural values play a significant role in shaping the development of technological advancements.\n\nIn my opinion, cultural values should influence technological advancements by prioritizing the needs and values of the community. For instance, in an agricultural setting, traditional farming practices may be valued over modern technology, but this doesn't mean that technology can't be incorporated to improve efficiency and productivity.\n\nSimilarly, in the finance sector, cultural values such as trust and security are essential when developing digital financial products. By understanding and respecting these values, we can create technologies that are not only innovative but also socially responsible.\n\nHowever, I also believe that cultural values should not limit technological advancements, but rather complement and guide them. By embracing cultural values such as sustainability, inclusivity, and social responsibility, we can create technologies that benefit not only individuals but also the community as a whole.\n\nAs someone who has organized events like Product Talk, I've seen firsthand the importance of communicating product progress and engaging with stakeholders to harness the power of innovations. By considering cultural values in technological advancements, we can create solutions that are not only effective but also meaningful and relevant to the community.\n\nUltimately, my goal is to contribute to the development of technologies that provide value to individuals and communities, and I believe that cultural values are essential in achieving this goal."
    },
    {
        "question" : "As a master’s student, what is the most challenging aspect of your studies so far?",
        "answer" : "As a master's student in Data Science and Artificial Intelligence, I would say that the most challenging aspect of my studies so far has been striking a balance between the theoretical foundations of the field and the practical application of the concepts. While I have a solid understanding of the mathematical and statistical underpinnings of data science, I have found it challenging to translate these concepts into real-world problems and solutions.\n\nAdditionally, the constant evolution of the field, with new technologies and techniques emerging regularly, has made it difficult for me to keep up with the latest developments and stay current with the latest research and advancements. However, this challenge has also been an opportunity for me to grow and learn, and I am excited to see how my knowledge and skills will evolve as I continue my studies.\n\nOverall, I believe that the most challenging aspect of my studies so far has been the need to bridge the gap between theoretical knowledge and practical application, and to stay adaptable and up-to-date with the rapidly changing landscape of data science and artificial intelligence."
    },
     {
        "question" : "What specific research interests or academic goals do you hope to achieve during your time as a master’s student?",
        "answer" : "As a Master's student in Data Science and Artificial Intelligence, I am eager to explore and develop research interests that align with my passion for providing value to clients and users in the technical industries. Specifically, I hope to achieve the following academic goals:\n\n1. **Develop expertise in machine learning applications**: I aim to delve deeper into the field of machine learning and explore its applications in various domains, such as education, finance, and agriculture, to identify areas where I can make a meaningful impact.\n\n2. **Investigate the use of AI in data-driven decision-making**: I am interested in investigating the role of Artificial Intelligence in data-driven decision-making, with a focus on developing predictive models and algorithms that can provide actionable insights for clients and users.\n\n3. **Explore the intersection of data science and social impact**: I hope to explore ways in which data science and AI can be used to drive positive social impact, such as improving agricultural productivity, promoting financial inclusion, or enhancing educational outcomes.\n\n4. **Collaborate with industry partners**: I aim to collaborate with industry partners to gain practical experience and insights into the challenges and opportunities facing technical industries, and to develop solutions that can be scaled and implemented in real-world settings.\n\nThrough these research interests and academic goals, I hope to gain a deeper understanding of the technical industries, develop cutting-edge research skills, and contribute to the development of digital products that provide value to clients and users."
    }
]
```
## Application
### Application Development
This Streamlit-based RAG (Retrieval-Augmented Generation) Chatbot enhances AI-generated responses by retrieving relevant context from documents before generating answers. It uses `llama-3.2-3b-preview` as the language model and FAISS for efficient document retrieval. Users can input questions, and the app fetches related content from an indexed document (e.g., a PDF), ensuring responses are context-aware and factually grounded. Optimizations like retrieval filtering, response length control, and efficient prompt structuring improve accuracy and performance.

### How to use web app
1) Download and run the App: `streamlit run app.py`
2) Enter a Question: (e.g., "How old are you?").
3) Document Retrieval: The app searches for relevant content from the indexed document using FAISS.
4) View the Results:
- The retrieved document excerpts (for reference).
- The AI-generated answer based on the retrieved context.

### Screenshots
![Webapp1](ss.png)
![Webapp2](ss.gif)
## Key Observations:
- Accurate Responses: The chatbot generates answers based on retrieved document content, improving factual accuracy.
- Efficient Retrieval: FAISS enables fast and relevant document searches, enhancing response quality.
- Optimized Performance: Using `llama-3.2-3b-preview` ensures faster inference while maintaining reasonable accuracy.
## Conclusion
This project successfully implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit, FAISS, and LLaMA-3. The chatbot enhances AI-generated responses by retrieving relevant document content before answering user queries. Optimizations in retrieval filtering, response length, and model tuning ensure improved accuracy and efficiency. This system demonstrates the power of AI-driven document-based question answering for real-world applications.
