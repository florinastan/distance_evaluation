# Distance evaluation
Computer program that evaluates automatically the distance between people in a pre-recorded video sequence


## Description
This project is a computer program that can measure the distance between people in a video with the help of a referance object. 

## Algorithm
- Resizing the video to a standard size
- Detect the people present in the video by using HOG detection
- Choosing the points we want to be tracked using a Haar Cascade model
- Follow the selected points thru the video with Lukas-Kanade algorithm
- Using the height of the leftmost person, calculate the distance between people

![ezgif com-gif-maker](https://user-images.githubusercontent.com/79220497/202676040-49c30515-84da-438f-b930-d4a65c48a7ad.gif)
