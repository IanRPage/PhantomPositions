# PhantomPositions

The advent of online job platforms like Indeed, Glassdoor, LinkedIn, and others
have given employers and job seekers greater access to top talent and
opportunities, respectively. On the same token, this wide access to job postings
presents a new attack vector for fraud. Bad actors can create enticing, but
deceptive job postings for the purpose of collecting sensitive applicant data
and even fees. This presents a threat to job seekers as it could lead to
financial loss, identity theft, or simply the emotional toll of getting their
hopes up.

We created PhantomPositions to address this problem using different data science
techniques, and to compare performance across each method in hopes of identifying
the most effective technique for detecting phantom job postings.

## Project Setup

### Environment and Dependencies

<details>
<summary>If using Windows 10 or Later</summary>
  
  <br>
  You may need to <strong>enable Long Paths</strong> to avoid issues with file
  paths during setup. You can do this by modifying your Registry.

  <a href ="https://www.youtube.com/watch?v=JIBsJx7U0Xw"> Watch this video for a
  step-by-step walkthrough on how to enable Long Paths in Windows 11</a>

</details>

After cloning the repository, run the following:
```
cd PhantomPositions
python -m venv venv
```

Then activate the virtual environment:

<details>
<summary>For Windows</summary>
  
```
.\venv\Scripts\activate
```
</details>

<details>
<summary>For Linux/macOS</summary>
  
```
source venv/bin/activate
```
</details>

From there, just install the dependencies from `requirements.txt`:
```
pip install -r requirements.txt
```

### Obtaining the Dataset

You need to download the EMSCAD dataset successfully run the notebooks. To do
so, run the `download_data` script:

<details>
<summary>For Windows</summary>
  
```
.\download_data.bat
```
</details>

<details>
<summary>For Linux/macOS</summary>
  
```
sh download_data.sh
```
</details>

### Verify Setup

After installing the dependencies and downloading the dataset, open
`environment_test.ipynb` in a notebook and run all the cells. This notebook
verifies that the environment is set up correctly.

## Running the Project

After setting up the environment correctly and activating it, you are ready to
run the notebooks! There are a few different notebooks to run, each
accomplishing different tasks.

- To explore the characteristics of the data run each cell in
  `src/dataExploration.ipynb`
  
- To check out the Random Forest, SVM, and Logistic Regression models
  individually, run `src/rf.ipynb`, `src/svm.ipynb`, and `src/lr.ipynb`,
  respectively.

    - Each notebook runs the model on the same preprocessing pipeline and then
      shows their confusion matrix, ROC curve, and Precision-Recall curve after
      fitting.

- To compare each model side-by-side, run `src/modelComparison.ipynb`.

    - It will show the same visualizations and metrics from the model notebooks,
      but combine them for easy comparison, in addition to more visualizations,
      and also save all the figures to the `outputs` directory.