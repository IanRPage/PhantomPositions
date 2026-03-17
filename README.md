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

After cloning this repository locally, copy these commands:
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
