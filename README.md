# 📊 Sentiment Analysis of Amazon Food Reviews using VADER and RoBERTa

This project performs sentiment analysis on Amazon food reviews using both VADER (rule-based) and RoBERTa (transformer-based deep learning). It generates interactive 2D scatter plots to visualize sentiment trends.

---

<img width="1470" alt="Image" src="https://github.com/user-attachments/assets/76dc16eb-fc34-4b1d-b690-0e1febbdb52d" />
<img width="1470" alt="Image" src="https://github.com/user-attachments/assets/37e69444-7978-4e30-a993-ee6e08771855" />
<img width="1470" alt="Image" src="https://github.com/user-attachments/assets/f47739bc-421a-46e0-ac35-d078479e521d" />


---

## ✅ **Step 1: Download the Dataset**

Download the dataset from Kaggle:

🔗 [Amazon Fine Food Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv)

* You will need a Kaggle account to download the dataset.
* After downloading, extract the file `Reviews.csv` and place it inside a folder named `amazon-data` in the root of your project directory.

Your project structure should look like this:

```
project-folder/
├── amazon-data/
│   └── Reviews.csv
├── Sentiment_Amazon.py
├── requirements.txt
```

---

## ✅ **Step 2: Set Up Python Environment**

It's recommended to use a virtual environment to avoid conflicts with other projects.

### **Create and Activate Virtual Environment**

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

## ✅ **Step 3: Install Required Packages**

Run the following commands inside your terminal:

```bash
python -m pip install -r requirements.txt
```

---

## ✅ **Step 4: Download NLTK Resources**

The code automatically downloads required NLTK resources. If you face issues, run manually:

```python
import nltk
nltk.download('vader_lexicon')
```

---

## ✅ **Step 5: Run the Sentiment Analysis**

Make sure your terminal is in the project directory and the virtual environment is activated.

Run:

```bash
python Sentiment_Amazon.py
```

The script will:

* Load the dataset (max 500 reviews).
* Perform sentiment analysis using:

  * **VADER** for rule-based sentiment scoring.
  * **RoBERTa** (via Hugging Face Transformers) for deep learning sentiment scoring.
* Generate an interactive 2D scatter plot using Plotly.

The plot shows **Negative vs Positive sentiment scores**, with hover information for each review.

---

## ✅ **Step 6: Adjust or Expand**

* The maximum reviews that can be viewed are **500**.
* To analyze fewer reviews, modify this line in your code:

```python
MAX_REVIEWS = 500  # Change the number as needed (max 500)
```

---

## ✅ **Optional: Virtual Environment Benefits**

Using a virtual environment with Python 3 prevents package conflicts and keeps project dependencies isolated.

---

# 🎉 You're all set! Run the script, explore the sentiment plots, and analyze Amazon food reviews!
