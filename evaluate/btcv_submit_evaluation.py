"""
This is the uploader script for all Sage Synapse MASI-hosted challenges.
You can find usage with --help and -h. Additionally, you can find more
informatin on our webpage: https://www.synapse.org/#!Synapse:syn3193805/wiki/
Pylint 9.8./10
"""
__author__ = "Stephen M. Damon"
__credits__ = ['Zhoubing Xu', 'Stephen Damon', 'Rob Harrigan',
               'Andrew Plassard', 'Bennett Landman']
__copyright__ = 'MASI Lab'
__version__ = '1.1.0'
__status__ = 'PRODUCTION'
__email__ = 'stephen.m.damon@vanderbilt.edu'
__purpose__ = 'Submit a folder to a specific entity for evaluation for the ' \
              '2015 Segmentation Outside the Cranial Vault Challenge'

import os
import sys
import synapseclient
from synapseclient import File
# from synapseclient.exceptions import SynapseHTTPError

CERVIX_STD = [3556505,3556507,3556509,3556511]
CERVIX_FREE = [3556513,3556515,3556517,3556519]
AB_STD = [3556479,3556481,3556483,3556485,3556487,3556489,3556491,3556493,3556495,3556497,3556499,3556501,3556503]
AB_FREE = [3556453,3556455,3556457,3556459,3556461,3556463,3556465,3556467,3556469,3556471,3556473,3556475,3556477]
def parse_args():
    """
    Parse the input arguments
    """
    from argparse import ArgumentParser, RawTextHelpFormatter
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-d', '--directory', dest='directory', required=True,
                        help='full path to the upload folder')
    parser.add_argument('-u', '--username', dest='username', required=True,
                        help='Your synapse username (probably your email)')
    parser.add_argument('-p', '--password', dest='password', required=True,
                        help='Your Synapse password')
    parser.add_argument('-t', '--team', dest='teamname', required=True,
                        help='Your team''s name as you would like it to appear'
                             ' on the leaderboard')
    parser.add_argument('-n', '--name', dest='name', required=True,
                        help='The name of your submission e.g., MASI_Majority_Vote'
                             ' on the leaderboard')
    parser.add_argument('-c', '--challenge', dest='challenge', required=True,
                        type=int, help='The challenge that you want to submit'
                                       ' to. Enter one of the following\n'
                                       '3257313: 2015 Abdomen Challenge'
                                       'Standard Challenge\n'
                                       '3381219: 2015 Abdomen Challenge'
                                       'Free Challenge\n'
                                       '3567563: 2015 Cervix Challenge'
                                       'Standard Challenge\n'
                                       '3381221: 2015 Cervix Challenge'
                                       'Free Challenge\n'
                                       '3260663: 2013 Canine Standard '
                                       'Challenge\n'
                                       '3567569: 2013 Canine Free Challenge\n'
                                       '3260659: 2013 Diencephalon Standard '
                                       'Challenge\n'
                                       '3567567: 2013 Diencephalon Free '
                                       'Challenge\n'
                                       '3260668: 2013 Cardiac Atlas Project '
                                       'Standard Challenge\n'
                                       '3567571: 2013 Cardiac Atlas Project '
                                       'Free Challenge')

    parser.add_argument('-l', '--launch', dest='launch', required=False,
                        default=False, action='store_true')
    return parser.parse_args()


def submit_to_challenge(evaluation, participant_file, sub_name):
    """
    Handle the submission process
    """
    SYN.submit(evaluation=evaluation,
               entity=participant_file,
               name=sub_name,
               teamName=ARGS.teamname)

    print("The number of your submission is %s" % sub_name)


if __name__ == '__main__':
    ARGS = parse_args()

    # Ensure that the file to upload does, in fact, exist
    if not os.path.exists(ARGS.directory):
        sys.exit('ERROR: Can not locate directory %s' % ARGS.directory)

    ID_DICT = {3257313: 'syn3249110',  # 2015 Abdomen Challenge STD
               3381219: 'syn3381225',  # 2015 Abdomen Challenge Free
               3567563: 'syn3249140',  # 2015 Cervix Challenge STD
               3381221: 'syn3381229',  # 2015 Cervix Challenge Free
               3260663: 'syn3270347',  # 2013 Canine Standard Challenge
               3567569: 'syn3270349',  # 2013 Canine Free Challenge
               3260659: 'syn3270351',  # 2013 Diencephalon Standard Challenge
               3567567: 'syn3270353',  # 2013 Diencephalon Free Challenge
               3260668: 'syn3270355',  # 2013 CAP STD Challenge
               3567571: 'syn3270357'}  # 2013 CAP Free Challenge

    if ARGS.challenge not in ID_DICT.keys():
        sys.exit('ERROR: The challenge type %i is not one of the following %s'
                 % (ARGS.challenge, ', '.join(str(X) for X in ID_DICT.keys())))

    # Login to synapse
    SYN = synapseclient.Synapse()
    try:
        SYN.login(ARGS.username, ARGS.password)
    except Exception as e:
        sys.exit('ERROR: Could not log in with credentials. '
                 'Please check username and password')

    PARTICIPANT_FILE = SYN.store(File(ARGS.directory,
                                      parent=ID_DICT[ARGS.challenge]))

    if ARGS.challenge in [3257313, 3381219, 3567563, 3381221]:
        # Submit to every leaderboard including the by-organ labels
        EVALUATION = SYN.getEvaluation(ARGS.challenge)
        submit_to_challenge(EVALUATION, PARTICIPANT_FILE, ARGS.name)

        if ARGS.challenge == 3257313:
            for EVAL in AB_STD:
                TEMP_EVAL = SYN.getEvaluation(EVAL)
                submit_to_challenge(TEMP_EVAL, PARTICIPANT_FILE, ARGS.name)
        elif ARGS.challenge == 3381219:
            for EVAL in AB_FREE:
                TEMP_EVAL = SYN.getEvaluation(EVAL)
                submit_to_challenge(TEMP_EVAL, PARTICIPANT_FILE, ARGS.name)
        elif ARGS.challenge == 3567563:
            for EVAL in CERVIX_STD:
                TEMP_EVAL = SYN.getEvaluation(EVAL)
                submit_to_challenge(TEMP_EVAL, PARTICIPANT_FILE, ARGS.name)
        elif ARGS.challenge == 3381221:
            for EVAL in CERVIX_FREE:
                TEMP_EVAL = SYN.getEvaluation(EVAL)
                submit_to_challenge(TEMP_EVAL, PARTICIPANT_FILE, ARGS.name)
    else:
        EVALUATION = SYN.getEvaluation(ARGS.challenge)
        submit_to_challenge(EVALUATION, PARTICIPANT_FILE, ARGS.name)


    # Log out
    SYN.logout()

