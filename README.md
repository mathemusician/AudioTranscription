# AudioTranscription


```Add subtitles to your video using machine learning!```

![tutorial-gif](./static/tutorial.gif)



# Implementation

The app can be split into three parts: 1) model 2) user interface 3) video-maker. For the model, I used a pytorch-lightning implementation of wav2vec2 to convert audio into text. For the user interface, I used streamlit. And for rendering the final video, I use ffmpeg.

