# 🖼️ Image-to-Story-to-Speech Streamlit App  

This Streamlit application demonstrates the power of **pre-trained models from Hugging Face** to create a pipeline that converts **images into descriptive text**, generates a **creative story**, and finally synthesizes the story into **speech audio**.  


## 🎯 **Overview**  
This project showcases how **image captioning**, **language generation**, and **text-to-speech models** can be seamlessly combined to deliver an interactive, AI-powered user experience. By leveraging state-of-the-art pre-trained models, this app serves as a practical example of multimodal AI in action.  



## 🛠️ **How the App Works**  

### 🔹 **1️⃣ Image-to-Text**  
- Users upload an image.  
- The **Salesforce/blip-image-captioning-base** model generates a textual description of the image.  

### 🔹 **2️⃣ Story Generation**  
- The image caption is passed to **meta-llama/Llama-3.2-1B** (via Hugging Face Inference API).  
- The model creates a creative story based on the caption and optional user prompts.  

### 🔹 **3️⃣ Text-to-Speech**  
- The story is converted into audio using the **microsoft/speecht5_tts** model.  
- Users can listen to the generated story directly in the app.  


## 🛠️ **Technical Stack**  

| **Component**       | **Technology/Model Used**                          |
|---------------------|----------------------------------------------------|
| **Frontend**        | **Streamlit**                                      |
| **Image Captioning**| **Salesforce/blip-image-captioning-base** (Hugging Face pipeline) |
| **Story Generation**| **meta-llama/Llama-3.2-1B** (Hugging Face Inference API) |
| **Text-to-Speech**  | **microsoft/speecht5_tts** (Hugging Face pipeline) |



## 🏃 **Getting Started**  

### 🔹 **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/camm93/image_to_audio_story_app.git
cd image_to_audio_story_app
````
### 🔹 **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
````

### 🔹 **3️⃣ Run the Streamlit App**
```bash
streamlit run app.py
````

### 🎨 **Example Workflow**

1. **Upload an Image**:  
   The app accepts images (e.g., `.jpg`, `.png`).  

2. **Image Captioning**:  
   **Input**:  
   ![Example Image](https://github.com/camm93/image_to_audio_story_app/blob/main/assets/man%20on%20island.png)

   **Caption Output**:  
"A child flying a kite in a sunny park."

3. **Story Generation**:  
**Prompt**:  
*"Write a fun story about the child and their kite."*  
**Story Output**:  
On a breezy afternoon, Alex soared their colorful kite high into the sky, dreaming of adventures...

4. **Text-to-Speech**:  
**Audio Output**:  
The app converts the story into natural-sounding speech that users can listen to directly in the app.  

---

### 💬 **Contact**
Feel free to reach out with questions or suggestions:  
- **Email**: crismur93@gmail.com
- **LinkedIn**: [Cristian's Profile](https://www.linkedin.com/in/cristianmurillom/)
