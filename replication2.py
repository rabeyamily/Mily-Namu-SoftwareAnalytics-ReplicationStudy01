#!/usr/bin/env python3
"""
Replication Study for RQ1 and RQ2

"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from github import Github, Auth
from github.GithubException import GithubException, UnknownObjectException
from datetime import timezone
import re
import random
import time
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# GitHub retry helper #todo: implement
def retry_github(fn, retries=5, base_delay=2):
    for attempt in range(retries):
        try:
            return fn()
        except GithubException as e:
            # for permanent client errors, fall back
            if e.status in (400, 401, 403, 404, 422):
                raise
            if attempt == retries - 1:
                raise
            sleep = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"GitHub error: {e}. Retrying in {sleep:.1f}s...")
            time.sleep(sleep)
        except Exception as e:
            if attempt == retries - 1:
                raise
            sleep = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"GitHub error: {e}. Retrying in {sleep:.1f}s...")
            time.sleep(sleep)

# Load environment variables from .env file
load_dotenv()

# Get GitHub token from environment
token = os.getenv("GITHUB_TOKEN")
if not token:
    print("Error: GITHUB_TOKEN not found in .env file")
    print("Please create a .env file with your GitHub token:")
    print("GITHUB_TOKEN=your_token_here")
    sys.exit(1)
auth = Auth.Token(token)
git_api = Github(auth=auth)
git_api.get_user().login

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data files should be in the same directory as this script
PR_DATA_FILE = os.path.join(SCRIPT_DIR, "pull_requests_meta_data.csv")
RELEASE_DATA_FILE = os.path.join(SCRIPT_DIR, "releases_meta_data.csv") 
PR_DATA_FILE_NEW = os.path.join(SCRIPT_DIR, "pull_requests_meta_data_new.csv")
RELEASE_DATA_FILE_NEW = os.path.join(SCRIPT_DIR, "releases_meta_data_new.csv") 

# Output directories (will be created in current directory)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
RESULTS_NEW_DIR = os.path.join(RESULTS_DIR, "data_collection")

ALPHA = 0.05  # Significance level
DELTA_THRESHOLDS = {'negligible': 0.147, 'small': 0.33, 'medium': 0.474}
PRACTICE_BEFORE_CI = "NO-CI"
PRACTICE_AFTER_CI = "CI"

# Create output directories
try:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_NEW_DIR, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create output directories: {e}")
    print(f"Results will be saved in: {SCRIPT_DIR}")
    RESULTS_DIR = SCRIPT_DIR
    FIGURES_DIR = SCRIPT_DIR
    RESULTS_NEW_DIR = SCRIPT_DIR

# List of projects with which to replicate data collection process
PR_DATA_COLLECTION = ["mizzy/serverspec", "andypetrella/spark-notebook", "driftyco/ng-cordova", "craftyjs/Crafty", "androidannotations/androidannotations"]

# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def cliffs_delta(group1, group2):
    """Calculate Cliff's Delta effect size."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    n1, n2 = len(group1), len(group2)
    greater = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    
    return (greater - less) / (n1 * n2)


def interpret_delta(delta):
    """Interpret Cliff's delta magnitude."""
    if np.isnan(delta):
        return 'undefined'
    abs_delta = abs(delta)
    if abs_delta < DELTA_THRESHOLDS['negligible']:
        return 'negligible'
    elif abs_delta < DELTA_THRESHOLDS['small']:
        return 'small'
    elif abs_delta < DELTA_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'large'


def mann_whitney_test(group1, group2):
    """Perform Mann-Whitney U test."""
    group1 = np.array(group1)[~np.isnan(group1)]
    group2 = np.array(group2)[~np.isnan(group2)]
    
    if len(group1) == 0 or len(group2) == 0:
        return np.nan, np.nan
    
    try:
        return mannwhitneyu(group1, group2, alternative='two-sided')
    except:
        return np.nan, np.nan


def wilcoxon_test(group1, group2):
    """Perform Wilcoxon signed-rank test."""
    group1 = np.array(group1)
    group2 = np.array(group2)
    mask = ~(np.isnan(group1) | np.isnan(group2))
    group1 = group1[mask]
    group2 = group2[mask]
    
    if len(group1) == 0:
        return np.nan, np.nan
    
    try:
        return wilcoxon(group1, group2, alternative='two-sided')
    except:
        return np.nan, np.nan


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(PR, RELEASE):
    """Load PR and release data."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Check if data files exist
    if not os.path.exists(PR):
        print(f"\nERROR: Cannot find pull_requests_meta_data.csv")
        print(f"Looking for: {PR}")
        print(f"\nPlease make sure the data files are in the same directory as this script:")
        print(f"  - pull_requests_meta_data.csv")
        print(f"  - releases_meta_data.csv")
        sys.exit(1)
    
    if not os.path.exists(RELEASE):
        print(f"\nERROR: Cannot find releases_meta_data.csv")
        print(f"Looking for: {RELEASE}")
        print(f"\nPlease make sure the data files are in the same directory as this script.")
        sys.exit(1)
    
    # Load PR data
    pr_df = pd.read_csv(PR)
    
    # Clean up column names
    if 'Unnamed: 0' in pr_df.columns:
        pr_df = pr_df.drop('Unnamed: 0', axis=1)
    if '' in pr_df.columns:
        pr_df = pr_df.drop('', axis=1)
    
    pr_df['lifetime'] = pr_df['merge_time'] + pr_df['delivery_time']
    pr_df['practice'] = pr_df['practice'].astype(str).str.strip()
    
    print(f"\nLoaded {len(pr_df):,} pull requests")
    print(f"  - {len(pr_df[pr_df['practice'] == PRACTICE_BEFORE_CI]):,} before CI")
    print(f"  - {len(pr_df[pr_df['practice'] == PRACTICE_AFTER_CI]):,} after CI")
    print(f"  - {pr_df['project'].nunique()} unique projects")
    
    # Load release data
    release_df = pd.read_csv(RELEASE)
    release_df['startedAt'] = pd.to_datetime(release_df['startedAt'])
    release_df['publishedAt'] = pd.to_datetime(release_df['publishedAt'])
    release_df['practice'] = release_df['practice'].astype(str).str.strip()
    
    print(f"\nLoaded {len(release_df):,} releases")
    print(f"  - {len(release_df[release_df['practice'] == PRACTICE_BEFORE_CI]):,} before CI")
    print(f"  - {len(release_df[release_df['practice'] == PRACTICE_AFTER_CI]):,} after CI")
    
    return pr_df, release_df


def get_projects_with_both_phases(df):
    """Get projects that have data in both CI and NO-CI phases."""
    projects_with_both = []
    for project in df['project'].unique():
        project_data = df[df['project'] == project]
        has_before = (project_data['practice'] == PRACTICE_BEFORE_CI).any()
        has_after = (project_data['practice'] == PRACTICE_AFTER_CI).any()
        if has_before and has_after:
            projects_with_both.append(project)
    return sorted(projects_with_both)


# ============================================================================
# RQ1 ANALYSIS
# ============================================================================

def analyze_rq1(pr_df, results_dir):
    """
    RQ1: Are merged pull requests released more quickly using continuous integration?
    Analyzes: delivery_time, merge_time, lifetime
    """
    print("\n" + "="*80)
    print("RQ1: Are merged pull requests released more quickly using continuous integration?")
    print("="*80)
    
    projects = get_projects_with_both_phases(pr_df)
    print(f"\nAnalyzing {len(projects)} projects with both phases...")
    
    metrics = ['delivery_time', 'merge_time', 'lifetime']
    all_results = {}
    
    for metric in metrics:
        results = []
        for project in projects:
            # Get data for this project
            before = pr_df[(pr_df['project'] == project) & 
                          (pr_df['practice'] == PRACTICE_BEFORE_CI)][metric].dropna().values
            after = pr_df[(pr_df['project'] == project) & 
                         (pr_df['practice'] == PRACTICE_AFTER_CI)][metric].dropna().values
            
            if len(before) == 0 or len(after) == 0:
                continue
            
            # Statistical tests
            _, p_value = mann_whitney_test(before, after)
            delta = cliffs_delta(before, after)
            
            results.append({
                'project': project,
                'metric': metric,
                'n_before': len(before),
                'n_after': len(after),
                'median_before': np.median(before),
                'median_after': np.median(after),
                'p_value': p_value,
                'cliffs_delta': delta,
                'effect_size': interpret_delta(delta),
                'significant': p_value < ALPHA if not np.isnan(p_value) else False,
                'faster_after_ci': delta > 0
            })
        
        results_df = pd.DataFrame(results)
        all_results[metric] = results_df
        
        # Print summary
        sig = results_df[results_df['significant']]
        n_sig = len(sig)
        faster_after = len(sig[sig['faster_after_ci']]) if n_sig > 0 else 0
        
        print(f"\nSummary for {metric}:")
        print(f"  Total projects: {len(results_df)}")
        print(f"  Significant (p < {ALPHA}): {n_sig} ({n_sig/len(results_df)*100:.1f}%)")
        if n_sig > 0:
            print(f"  Faster AFTER CI: {faster_after}/{n_sig} ({faster_after/n_sig*100:.1f}%)")
            print(f"  Median Cliff's delta: {results_df['cliffs_delta'].median():.3f}")
    
    # Save results
    for metric, df in all_results.items():
        filepath = os.path.join(results_dir, f'rq1_{metric}_results.csv')
        df.to_csv(filepath, index=False)
    
    # Create summary
    summary = []
    for metric, df in all_results.items():
        sig = df[df['significant']]
        n_sig = len(sig)
        faster_after = len(sig[sig['faster_after_ci']]) if n_sig > 0 else 0
        summary.append({
            'Metric': metric,
            'Total': len(df),
            'Significant': n_sig,
            'Faster After CI': f"{faster_after}/{n_sig}" if n_sig > 0 else "N/A",
            'Percent Faster After': f"{faster_after/n_sig*100:.1f}%" if n_sig > 0 else "N/A",
            'Median Delta': f"{df['cliffs_delta'].median():.3f}"
        })
    
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(results_dir, 'rq1_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("RQ1 SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    return all_results


# ============================================================================
# RQ2 ANALYSIS
# ============================================================================

def analyze_rq2(release_df, results_dir):
    """
    RQ2: Does increased development activity after CI increase delivery time?
    Analyzes: PR rates, release frequency, churn
    """
    print("\n" + "="*80)
    print("RQ2: Does increased development activity after CI increase delivery time?")
    print("="*80)
    
    projects = get_projects_with_both_phases(release_df)
    print(f"\nAnalyzing {len(projects)} projects...")
    
    # Metrics to analyze
    metrics = {
        'created_pull_requests': 'Submitted PRs per release',
        'merged_pull_requests': 'Merged PRs per release',
        'released_pull_requests': 'Delivered PRs per release',
        'sum_submitted_pr_churn': 'Code churn per release'
    }
    
    all_results = {}
    
    # Analyze each metric
    for metric, description in metrics.items():
        results = []
        for project in projects:
            before = release_df[(release_df['project'] == project) & 
                               (release_df['practice'] == PRACTICE_BEFORE_CI)][metric].dropna().values
            after = release_df[(release_df['project'] == project) & 
                              (release_df['practice'] == PRACTICE_AFTER_CI)][metric].dropna().values
            
            if len(before) == 0 or len(after) == 0:
                continue
            
            results.append({
                'project': project,
                'median_before': np.median(before),
                'median_after': np.median(after)
            })
        
        all_results[metric] = pd.DataFrame(results)
    
    # Release frequency
    freq_results = []
    for project in projects:
        for practice in [PRACTICE_BEFORE_CI, PRACTICE_AFTER_CI]:
            proj_releases = release_df[(release_df['project'] == project) & 
                                      (release_df['practice'] == practice)]
            if len(proj_releases) == 0:
                continue
            
            start = proj_releases['startedAt'].min()
            end = proj_releases['publishedAt'].max()
            years = (end - start).total_seconds() / (365.25 * 24 * 3600)
            
            if years > 0:
                if practice == PRACTICE_BEFORE_CI:
                    freq_results.append({'project': project, 'before': len(proj_releases) / years})
                else:
                    if freq_results and freq_results[-1]['project'] == project:
                        freq_results[-1]['after'] = len(proj_releases) / years
    
    freq_df = pd.DataFrame([r for r in freq_results if 'after' in r])
    all_results['release_frequency'] = freq_df
    
    # Aggregate analysis
    aggregate = []
    
    for metric, description in metrics.items():
        df = all_results[metric]
        before = df['median_before'].values
        after = df['median_after'].values
        
        _, p_val = wilcoxon_test(before, after)
        delta = cliffs_delta(before, after)
        
        increased = np.sum(after > before)
        
        aggregate.append({
            'Metric': description,
            'Median Before': f"{np.median(before):.1f}",
            'Median After': f"{np.median(after):.1f}",
            'Projects Increased': f"{increased}/{len(before)} ({increased/len(before)*100:.1f}%)",
            'p-value': f"{p_val:.6f}" if not np.isnan(p_val) else "N/A",
            'Significant': 'Yes' if p_val < ALPHA else 'No',
            'Cliff\'s Delta': f"{delta:.3f}",
            'Effect': interpret_delta(delta)
        })
    
    # Release frequency
    before = freq_df['before'].values
    after = freq_df['after'].values
    _, p_val = wilcoxon_test(before, after)
    delta = cliffs_delta(before, after)
    increased = np.sum(after > before)
    
    aggregate.append({
        'Metric': 'Releases per year',
        'Median Before': f"{np.median(before):.2f}",
        'Median After': f"{np.median(after):.2f}",
        'Projects Increased': f"{increased}/{len(before)} ({increased/len(before)*100:.1f}%)",
        'p-value': f"{p_val:.6f}" if not np.isnan(p_val) else "N/A",
        'Significant': 'Yes' if p_val < ALPHA else 'No',
        'Cliff\'s Delta': f"{delta:.3f}",
        'Effect': interpret_delta(delta)
    })
    
    aggregate_df = pd.DataFrame(aggregate)
    agg_path = os.path.join(results_dir, 'rq2_aggregate_results.csv')
    aggregate_df.to_csv(agg_path, index=False)
    
    print("\n" + "="*80)
    print("RQ2 AGGREGATE RESULTS")
    print("="*80)
    print(aggregate_df.to_string(index=False))
    
    # Save individual results
    for metric, df in all_results.items():
        filepath = os.path.join(results_dir, f'rq2_{metric}_results.csv')
        df.to_csv(filepath, index=False)
    
    return all_results, aggregate_df

# ============================================================================
# Data Collection
# ============================================================================

def data_collection() :
    print("Starting data collection...")

    PR_metadata = {
        "":[], "X":[], "project":[], "language":[], "pull_id":[], "pull_number":[], "commits_per_pr":[], "changed_files":[], 
        "churn":[], "comments":[], "comments_interval":[], "merge_workload":[], "description_length":[], 
        "contributor_experience":[], "queue_rank":[], "contributor_integration":[], "stacktrace_attached":[],
        "activities":[], "merge_time":[], "delivery_time":[], "practice":[]
    }
    releases_metadata = {
        "project":[], "title":[], "startedAt":[], "publishedAt":[], "release_duration":[], "created_pull_requests": [], 
        "merged_pull_requests": [], "released_pull_requests": [], "sum_submitted_pr_churn": [], "practice": []
    }

    for pr_name in PR_DATA_COLLECTION :
        repo = retry_github(lambda: git_api.get_repo(pr_name))
        languages = retry_github(lambda: repo.get_languages())
        lang_tmp, num_tmp = next(iter(languages.items()))
        for lang, num in languages.items() :
            if num > num_tmp :
                lang_tmp, num_tmp = lang, num
        language = lang_tmp

        pulls = list(retry_github(lambda: repo.get_pulls(state='all', sort="created", direction="desc")))
        pulls_open = list(retry_github(lambda: repo.get_pulls(state="open", sort="created", direction="desc")))
        pulls_release = list(retry_github(lambda: repo.get_pulls(state="closed", sort="updated", direction="desc")))

        releases = [r for r in retry_github(lambda: repo.get_releases()) if not r.prerelease]

        pr_ctr = 0

        for pr in pulls :
            commit = None
            sha = pr.merge_commit_sha or pr.head.sha
            if sha:
                try:
                    commit = retry_github(lambda: repo.get_commit(sha))
                except GithubException as e:
                    if e.status == 422 and pr.merge_commit_sha and pr.head.sha:
                        # fallback to head sha
                        try:
                            commit = retry_github(lambda: repo.get_commit(pr.head.sha))
                        except Exception:
                            commit = None
                    else:
                        commit = None
            
            pr_ctr += 1
            collect_PR_metadata(PR_metadata, language, pr_name, pulls, pulls_open, pulls_release, releases, pr_ctr, pr, commit)
            #print("pr_ctr = ", pr_ctr)
        releases.sort(key=lambda r: r.published_at)
        collect_releases_metadata(repo, releases_metadata, pr_name, pulls, releases)



    df_PR = pd.DataFrame(PR_metadata)
    df_PR.to_csv(PR_DATA_FILE_NEW, index=False)

    df_release = pd.DataFrame(releases_metadata)
    df_release.to_csv(RELEASE_DATA_FILE_NEW, index=False)


def collect_PR_metadata(PR_metadata, language, pr_name, pulls, pulls_open, pulls_release, releases, pr_ctr, pr, commit) :
    PR_metadata[""].append(pr_ctr)
    PR_metadata["X"].append(pr_ctr)
    PR_metadata["project"].append(pr_name)
    PR_metadata["language"].append(language)
    PR_metadata["pull_id"].append(pr.id)
    PR_metadata["pull_number"].append(pr.number)
    PR_metadata["commits_per_pr"].append(pr.commits)
    PR_metadata["changed_files"].append(pr.changed_files)
    PR_metadata["churn"].append(pr.additions + pr.deletions)
    comments = pr.get_comments()
    PR_metadata["comments"].append(comments.totalCount)
    PR_metadata["comments_interval"].append(get_interval_average(comments))
    PR_metadata["merge_workload"].append(get_merge_workload(pulls_open, pr))
    if not pr.body :
        PR_metadata["description_length"].append(0)
        PR_metadata["stacktrace_attached"].append(0)
    else :
        PR_metadata["description_length"].append(len(pr.body))
        body = pr.body.lower()
        if "traceback" in body or "exception" in body or "error" in body :
            PR_metadata["stacktrace_attached"].append(1)
        else :
            PR_metadata["stacktrace_attached"].append(0)
    PR_metadata["contributor_experience"].append(get_contributor_experience(pulls, pr))
    PR_metadata["queue_rank"].append(get_queue_rank(pulls_release, pr))
    PR_metadata["contributor_integration"].append(get_average_delivery_time(pulls, pr))
    total_entries = (
        retry_github(lambda: pr.get_commits()).totalCount +
        retry_github(lambda: pr.get_issue_events()).totalCount
    )
    PR_metadata["activities"].append(total_entries)
    PR_metadata["merge_time"].append(get_merge_time_days(pr))
    PR_metadata["delivery_time"].append(get_delivery_time_days(pr, releases))
    PR_metadata["practice"].append(check_CI(commit))


#Convert a naive datetime from GitHub to UTC-aware datetime
def to_utc_aware(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def get_interval_average(comments) :
    if(comments.totalCount <= 0) :
        return 0
    interval_tmp = to_utc_aware(comments[0].created_at).timestamp()
    interval_total = 0
    for comment in comments :
        interval_total += to_utc_aware(comment.created_at).timestamp() - interval_tmp
        interval_tmp = to_utc_aware(comment.created_at).timestamp()
    interval_average = interval_total / comments.totalCount
    return interval_average

def get_merge_workload(pulls_open, pr):
    pr_created_at = to_utc_aware(pr.created_at)
    
    # Filter PRs created before this PR
    prior_open_prs = [
        other_pr for other_pr in pulls_open
        if to_utc_aware(other_pr.created_at) < pr_created_at
    ]
    
    return len(prior_open_prs)

#Count the number of PRs previously submitted by the author of 'pr'
def get_contributor_experience(pulls, pr):
    author = pr.user.login
    pr_created_at = to_utc_aware(pr.created_at)
    
    # Count PRs by the same author created before this PR
    prior_prs_by_author = [
        p for p in pulls
        if to_utc_aware(p.created_at) < pr_created_at and p.user.login == author
    ]
    
    return len(prior_prs_by_author)

#Compute the queue position of a PR among all merged PRs in the release
def get_queue_rank(pulls_release, pr):
    merged_prs = [p for p in pulls_release if p.merged_at is not None]
    merged_prs.sort(key=lambda p: to_utc_aware(p.merged_at))

    for idx, p in enumerate(merged_prs, start=1):
        if p.id == pr.id:
            return idx
    
    # PR not found in merged PRs
    return 0

#Compute the average delivery time (in days) of previously merged PRs
def get_average_delivery_time(pulls, pr):
    author = pr.user.login
    pr_created_at = to_utc_aware(pr.created_at)

    # Filter PRs: previously submitted by the author and merged
    prior_merged_prs = [
        p for p in pulls
        if p.user.login == author
        and to_utc_aware(p.created_at) < pr_created_at
        and p.merged_at is not None
    ]

    if not prior_merged_prs:
        return 0  # No previous merged PRs

    # Compute time to merge for each prior PR
    time_deltas = [
        (to_utc_aware(p.merged_at) - to_utc_aware(p.created_at)).total_seconds()
        for p in prior_merged_prs
    ]

    # Average in days
    avg_seconds = sum(time_deltas) / len(time_deltas)
    avg_days = avg_seconds / (24 * 3600)

    return avg_days

#Compute the number of days between PR submission and merge
def get_merge_time_days(pr):
    if pr.merged_at is None:
        return 0
    created = to_utc_aware(pr.created_at)
    merged = to_utc_aware(pr.merged_at)
    delta = merged - created

    return delta.total_seconds() / (24 * 3600)  # convert seconds to days

#Calculate delivery time in days

def get_delivery_time_days(pr, releases):
    
    if pr.merged_at is None or releases is None:
        return 0

    merged = pr.merged_at
    release_date = releases[0].published_at
    
    for release in releases :
        if release.published_at < merged :
            break
        release_date = release.published_at

    delta = release_date - merged

    return delta.total_seconds() / (24 * 3600)  # convert seconds to days


def check_CI(commit=None) :
    if commit==None :
        return "NO-CI"

    # Combined status
    combined_status = retry_github(lambda: commit.get_combined_status())
    if combined_status.total_count > 0:
        return "CI"

    # Check suites (modern GitHub Actions and other apps)
    check_suites = retry_github(lambda: commit.get_check_suites())
    if len(list(check_suites)) > 0:
        return "CI"
    
    return "NO-CI"



def collect_releases_metadata(repo, releases_metadata, pr_name, pulls, releases) :
    pattern = re.compile(r"Merge pull request #(\d+)")

    for i, release in enumerate(releases):
        published_at = release.published_at
        started_at = releases[i-1].published_at if i > 0 else published_at

        release_duration = (published_at - started_at).days

        # PRs in this release period
        prs_in_period = [
            pr for pr in pulls
            if pr.created_at >= started_at and pr.created_at <= published_at
        ]
        merged_prs_in_period = [
            pr for pr in prs_in_period
            if pr.merged_at and started_at <= pr.merged_at <= published_at
        ]

        # Identify released PRs via commits in release tag
        try:
            #commit = retry_github(lambda: repo.get_commit(release.tag.commit.sha))
            tag_sha = get_tag_commit_sha(repo, release.tag_name)
            commit = retry_github(lambda: repo.get_commit(tag_sha))
            # Compare with previous release to get commits in this release
            if i > 0:
                compare = retry_github(lambda: repo.compare(releases[i-1].tag_name, release.tag_name))
                commits_in_release = compare.commits
            else:
                commits_in_release = [commit]
        except Exception:
            commits_in_release = []

        released_pr_numbers = set()
        sum_churn = 0

        for c in commits_in_release:
            msg = c.commit.message
            match = pattern.search(msg)
            if match:
                released_pr_numbers.add(int(match.group(1)))
                sum_churn += c.stats.additions + c.stats.deletions

        # Count PRs released
        released_prs = [pr for pr in merged_prs_in_period if pr.number in released_pr_numbers]

        releases_metadata["project"].append(pr_name)
        releases_metadata["title"].append(release.tag_name)
        releases_metadata["startedAt"].append(started_at.strftime("%Y-%m-%d %H:%M:%S"))
        releases_metadata["publishedAt"].append(published_at.strftime("%Y-%m-%d %H:%M:%S"))
        releases_metadata["release_duration"].append(release_duration)
        releases_metadata["created_pull_requests"].append(len(prs_in_period))
        releases_metadata["merged_pull_requests"].append(len(merged_prs_in_period))
        releases_metadata["released_pull_requests"].append(len(released_prs))
        releases_metadata["sum_submitted_pr_churn"].append(sum_churn)
        releases_metadata["practice"].append(check_release_CI(commits_in_release))

def check_release_CI(commits_in_release) :
    if not commits_in_release :
        return "NO-CI"

    for c in commits_in_release:
        if check_CI(c) == "CI" :
            return "CI"
        
    return "NO-CI"

def get_tag_commit_sha(repo, tag_name):
    # Resolve a release tag name to a commit SHA
    
    ref = retry_github(lambda: repo.get_git_ref(f"tags/{tag_name}"))
    obj = ref.object  # GitObject

    # annotated tag => object points to a tag object; dereference once
    if obj.type == "tag":
        tag_obj = retry_github(lambda: repo.get_git_tag(obj.sha))
        return tag_obj.object.sha

    # lightweight tag => object points directly to commit
    return obj.sha



# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_visualizations(rq1_results, rq2_results):
    """Create key visualizations."""
    # RQ1 Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, title) in enumerate([
        ('delivery_time', 'Delivery Time'),
        ('merge_time', 'Merge Time'),
        ('lifetime', 'Lifetime')
    ]):
        df = rq1_results[metric]
        data = pd.DataFrame({
            'Value': list(df['median_before']) + list(df['median_after']),
            'Phase': ['NO-CI'] * len(df) + ['CI'] * len(df)
        })
        
        sns.boxplot(x='Phase', y='Value', data=data, ax=axes[idx],
                   palette=['#FF6B6B', '#4ECDC4'])
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylabel('Days')
        axes[idx].set_xlabel('')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'rq1_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # RQ2 Summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_plot = [
        ('created_pull_requests', 'Submitted PRs', (0, 0)),
        ('merged_pull_requests', 'Merged PRs', (0, 1)),
        ('released_pull_requests', 'Delivered PRs', (1, 0)),
        ('sum_submitted_pr_churn', 'Code Churn', (1, 1))
    ]
    
    for metric, title, (row, col) in metrics_plot:
        df = rq2_results[metric]
        data = pd.DataFrame({
            'Value': list(df['median_before']) + list(df['median_after']),
            'Phase': ['NO-CI'] * len(df) + ['CI'] * len(df)
        })
        
        sns.boxplot(x='Phase', y='Value', data=data, ax=axes[row, col],
                   palette=['#FF6B6B', '#4ECDC4'])
        axes[row, col].set_title(title, fontweight='bold')
        axes[row, col].set_ylabel('Count per Release')
        axes[row, col].set_xlabel('')
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'rq2_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print(" CI REPLICATION STUDY - RQ1 & RQ2")
    print(" Bernardo et al. (2018)")
    print("="*80 + "\n")
    
    # Load data
    pr_df, release_df = load_data(PR_DATA_FILE, RELEASE_DATA_FILE)
    
    # RQ1 Analysis
    rq1_results = analyze_rq1(pr_df, RESULTS_DIR)
    
    # RQ2 Analysis
    rq2_results, rq2_aggregate = analyze_rq2(release_df, RESULTS_DIR)
    
    # Visualizations
    create_visualizations(rq1_results, rq2_results)
    
    # Summary
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n" + "="*80)
    print(" KEY FINDINGS")
    print("="*80)
    
    # RQ1 findings
    print("\nRQ1: Impact on PR Delivery")
    for metric in ['delivery_time', 'merge_time', 'lifetime']:
        df = rq1_results[metric]
        sig = df[df['significant']]
        if len(sig) > 0:
            faster_after = len(sig[sig['faster_after_ci']])
            print(f"  {metric}: {faster_after}/{len(sig)} ({faster_after/len(sig)*100:.1f}%) faster after CI")
    
    # RQ2 findings
    print("\nRQ2: Development Activity Changes")
    for _, row in rq2_aggregate.iterrows():
        print(f"  {row['Metric']}: {row['Median Before']} → {row['Median After']}")

    # Data collection replication
    print("\n" + "="*80)
    print(" REPRODUCTION (DATA COLLECTION REPLICATION)")
    print("="*80)

    data_collection()

    # Load the newly created data
    pr_df_new, release_df_new = load_data(PR_DATA_FILE_NEW, RELEASE_DATA_FILE_NEW)

    # RQ1 Analysis
    rq1_results_new = analyze_rq1(pr_df_new, RESULTS_NEW_DIR)

    # Final summary
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n" + "="*80)
    print(" KEY FINDINGS")
    print("="*80)
    
    # RQ1 findings
    print("\nRQ1: Impact on PR Delivery (Reproduced)")
    for metric in ['delivery_time', 'merge_time', 'lifetime']:
        df = rq1_results_new[metric]
        sig = df[df['significant']]
        if len(sig) > 0:
            faster_after = len(sig[sig['faster_after_ci']])
            print(f"  {metric}: {faster_after}/{len(sig)} ({faster_after/len(sig)*100:.1f}%) faster after CI")

    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()