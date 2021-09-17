import json
import requests
import csv
​
# Developer's token from github, generate it and make sure it has read/write access
# Settings > Developer settings > Personal access tokens
TOKEN = 'ghp_your_token'
​
# The repository to add this issue to
REPO_NAME = 'incore-services'
​
# 1 represents 1.6.0 milestone created in github: https://github.com/IN-CORE/incore-services/milestones
MILESTONE = 1
​
# The repo should have a "new feature" tag created first
TASK_MAPPING = {
    'Epic': "new feature",
    'New Feature': "new feature",
    'Improvement': "enhancement",
    "Story": "enhancement",
    "Task": "documentation",
    "Bug": "bug"
}
​
​
def create_github_issue(title, body=None, milestone=None, issue_type=None):
    url = 'https://api.github.com/repos/IN-CORE/%s/issues' % REPO_NAME
​
headers = {
    "Authorization": "token %s" % TOKEN,
    'Content-Type': 'application/json',
    'Accept': 'application/vnd.github.v3+json'
}
​
labels = [TASK_MAPPING[issue_type]]
# Create our issue
issue = {'title': title,
         'body': body,
         'milestone': milestone,
         'labels': labels}
​
r = requests.post(url=url, headers=headers, data=json.dumps(issue))
print(r.content)
​
​
if __name__ == '__main__':
    with open('issues.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            create_github_issue(row['Summary'], row['Description'], milestone=MILESTONE, issue_type=row['Issue Type'])