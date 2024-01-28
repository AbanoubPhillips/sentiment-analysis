import googleapiclient.discovery
import pandas as pd

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyDgUYgmr0Q3x4HdRQhp2_kdd6EFrRG96DA"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

comments = []

def getitems(response):
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['likeCount'],
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

    df = pd.DataFrame(comments, columns=['author', 'like_count', 'text','video_id','public'])
    return df

    for i in ['QOcP5OvSwlI','Lfzu74XDyco','TiS6vnju_mI','cYwioeHu_OU']:
        df = getcomments(i)

    df.to_csv('youtube_multi_videos_comments.csv')
    print(df["video_id"].value_counts())