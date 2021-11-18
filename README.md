# AudioTranscription


```Add subtitles to your video using machine learning!```

![tutorial-gif](./static/tutorial.gif)

# Web App [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourGitHubName/yourRepo/yourApp/)

[Link to web application](https://share.streamlit.io/mathemusician/audiotranscription/main/main.py)


# Desktop Version

1) Download repository
`Hi`

2) Install python requirements (Tested mainly on Python 3.9.0)
`pip install -r requirements.txt`

3) Install ffmpeg
*Using Homebrew on Mac*
`brew install ffmpeg`
*Windows*
Install from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

# Implementation

The app can be split into three parts: 1) model 2) user interface 3) video-maker. For the model, I used a pytorch-lightning implementation of wav2vec2 to convert audio into text. For the user interface, I used streamlit. And for rendering the final video, I use ffmpeg.

