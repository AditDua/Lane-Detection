from pytube import YouTube


def download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")


yt_link = input("Enter the YouTube video URL: ")
download(yt_link)
