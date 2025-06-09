from io import StringIO
import time
import subprocess
import pandas as pd


def get_latest_submission(competition: str) -> pd.Series:
    """ Retrieve information on the latest submission to a competition from kaggle.
    Args:
        competition (str): Name of the competition.
    Returns:
        pd.Series: First line of the table returned by kaggle.
    """
    result = subprocess.run(['kaggle', 'competitions', 'submissions', '-c', competition],
                            capture_output=True, text=True, check=False)
    output = result.stdout
    df = pd.read_csv(StringIO(output), sep=r'\s{2,}', engine='python')

    return df.iloc[1]


def submit_prediction(competition: str, csv_file: str, message: str, timeout: float = 300,
                      interval: float = 10) -> float:
    """ Submit a prediction csv file to kaggle, wait for success and return the public LB score for
        the submission.
    Args:
        competition (str): Name of the competition.
        csv_file (str): Path to the csv file.
        message (str): Submit message for kaggle.
        timeout (float, optional): Maximum time to wait for the submission to succeed. Defaults to
            300.
        interval (float, optional): Time between attempts. Defaults to 10.
    Raises:
        RuntimeError: Submission failed.
        TimeoutError: Submission did not succeed within timeout duration.
    Returns:
        float: Public score for the submission on the leader board.
    """
    # submit
    _ = subprocess.run(['kaggle', 'competitions', 'submit',
                        '-c', competition,
                        '-f', csv_file,
                        '-m', message
                        ],
                        check=False)

    # retrieve public score
    start = time.time()
    while time.time() - start < timeout:
        submission = get_latest_submission(competition)
        status = submission['status'].lower()
        if status == 'submissionstatus.complete':
            return submission['publicScore']

        if status == 'error':
            raise RuntimeError('Submission failed with error.')

        time.sleep(interval)

    raise TimeoutError('Timed out waiting for submission to complete.')
