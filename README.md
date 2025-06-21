# Handwritten Digit Generation Web App (MNIST)

This project contains the code for training a Deep Convolutional Generative Adversarial Network (DCGAN) to generate handwritten digits and a Streamlit web application to display the results.

## Project Structure

- `train.py`: The script for training the DCGAN model on the MNIST dataset.
- `app.py`: The Streamlit web application for generating and displaying digits.
- `generator.pth`: The trained weights of the generator model (you need to generate this by running `train.py`).
- `requirements.txt`: The list of Python dependencies.
- `README.md`: This file.

## How to Run

### 1. Training the Model (on Google Colab)

The model is designed to be trained on a GPU. Google Colab provides free T4 GPU access, which is sufficient for this task.

1.  **Open Google Colab:** Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Upload the training script:** Create a new notebook and upload the content of `train.py`.
3.  **Run the training:** Execute the cells in the notebook. The training process will take about 20-30 minutes. It will print the loss for the generator and discriminator at the end of each epoch.
4.  **Download the model weights:** After the training is complete, the file `generator.pth` will be saved in the Colab environment. Download this file to your local machine.

### 2. Running the Web App Locally

To test the application on your local machine before deploying:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Place the model weights:** Put the downloaded `generator.pth` file in the root of the project directory.
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    Your browser should open a new tab with the running application.

### 3. Deploying to Streamlit Community Cloud

Streamlit Community Cloud is a free service for deploying Streamlit apps.

1.  **Create a GitHub Repository:** Push your project to a new GitHub repository. Make sure to include:
    - `app.py`
    - `requirements.txt`
    - `generator.pth` (You can use Git LFS for larger files, but this model should be small enough for a standard repository).
2.  **Sign up for Streamlit Community Cloud:** If you don't have an account, sign up at [streamlit.io/cloud](https://streamlit.io/cloud).
3.  **Deploy the app:**
    - Click "New app".
    - Connect your GitHub account and select the repository you just created.
    - The branch should be `main` (or your default branch) and the main file path should be `app.py`.
    - Click "Deploy!". Streamlit will handle the rest.

Your application will be live at a public URL. 