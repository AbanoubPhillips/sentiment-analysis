import googleapiclient.discovery
import pandas as pd

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyBDJBeKx7_8cEMDARCYUcncUXTEMd7dr0g"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

comments = []

def getitems(response):
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['textOriginal'],
            comment['videoId'],
            public
        ])

def getcomments(video):
    request = youtube.commentThreads().list(
    part="snippet",
    videoId=video,
    maxResults=100
    )
    # Execute the request.
    response = request.execute()
    getitems(response)

    while (1 == 1):
        try:
            nextPageToken = response['nextPageToken']
        except KeyError:
            break
        nextPageToken = response['nextPageToken']
        # Create a new request object with the next page token.
        nextRequest = youtube.commentThreads().list(part="snippet", videoId=video, maxResults=100, pageToken=nextPageToken)
        # Execute the next request.
        response = nextRequest.execute()
        # Get the comments from the next response.
        getitems(response)

    df = pd.DataFrame(comments, columns=['author','text','video_id','public'])
    return df

video_ids = pd.read_excel('video_id.xlsx')
for i in video_ids['video_id']:
    df = getcomments(i)

# save data to csv file
df.to_csv('youtube_multi_videos_comments.csv', escapechar='\\')
