"""GitHub integration for repository analysis and CI/CD integration."""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class GitHubRepository:
    """GitHub repository information."""
    name: str
    full_name: str
    url: str
    default_branch: str
    description: Optional[str] = None


@dataclass
class PullRequest:
    """Pull request information."""
    number: int
    title: str
    state: str
    head_sha: str
    base_branch: str
    url: str


class GitHubClient:
    """Client for GitHub API integration."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub client.
        
        Args:
            token: GitHub API token (uses GITHUB_TOKEN env var if None)
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "formal-circuits-gpt/1.0"
        }
        
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def get_repository_info(self, owner: str, repo: str) -> Optional[GitHubRepository]:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository information or None if not found
        """
        try:
            import requests
            
            url = f"{self.base_url}/repos/{owner}/{repo}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return GitHubRepository(
                    name=data["name"],
                    full_name=data["full_name"],
                    url=data["html_url"],
                    default_branch=data["default_branch"],
                    description=data.get("description")
                )
            
            return None
            
        except ImportError:
            raise ImportError("requests library required for GitHub integration")
        except Exception as e:
            print(f"GitHub API error: {e}")
            return None
    
    def create_issue(self, owner: str, repo: str, title: str, body: str, 
                    labels: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Create an issue in the repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: Issue labels
            
        Returns:
            Created issue data or None if failed
        """
        if not self.token:
            print("GitHub token required for creating issues")
            return None
        
        try:
            import requests
            
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            data = {
                "title": title,
                "body": body
            }
            
            if labels:
                data["labels"] = labels
            
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                return response.json()
            else:
                print(f"Failed to create issue: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error creating issue: {e}")
            return None
    
    def get_pull_requests(self, owner: str, repo: str, 
                         state: str = "open") -> List[PullRequest]:
        """Get pull requests for repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: PR state (open, closed, all)
            
        Returns:
            List of pull requests
        """
        try:
            import requests
            
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
            params = {"state": state}
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                prs = []
                for pr_data in response.json():
                    pr = PullRequest(
                        number=pr_data["number"],
                        title=pr_data["title"],
                        state=pr_data["state"],
                        head_sha=pr_data["head"]["sha"],
                        base_branch=pr_data["base"]["ref"],
                        url=pr_data["html_url"]
                    )
                    prs.append(pr)
                return prs
            
            return []
            
        except Exception as e:
            print(f"Error getting pull requests: {e}")
            return []
    
    def set_commit_status(self, owner: str, repo: str, sha: str, 
                         state: str, description: str, 
                         context: str = "formal-circuits-gpt/verification") -> bool:
        """Set commit status for verification results.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            state: Status state (pending, success, failure, error)
            description: Status description
            context: Status context
            
        Returns:
            True if status was set successfully
        """
        if not self.token:
            print("GitHub token required for setting commit status")
            return False
        
        try:
            import requests
            
            url = f"{self.base_url}/repos/{owner}/{repo}/statuses/{sha}"
            data = {
                "state": state,
                "description": description,
                "context": context
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            return response.status_code == 201
            
        except Exception as e:
            print(f"Error setting commit status: {e}")
            return False
    
    def create_workflow_dispatch(self, owner: str, repo: str, workflow_id: str, 
                                ref: str = "main", inputs: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger workflow dispatch event.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Workflow ID or filename
            ref: Git reference to run workflow on
            inputs: Workflow inputs
            
        Returns:
            True if workflow was triggered successfully
        """
        if not self.token:
            print("GitHub token required for workflow dispatch")
            return False
        
        try:
            import requests
            
            url = f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
            data = {
                "ref": ref
            }
            
            if inputs:
                data["inputs"] = inputs
            
            response = requests.post(url, headers=self.headers, json=data)
            return response.status_code == 204
            
        except Exception as e:
            print(f"Error triggering workflow: {e}")
            return False