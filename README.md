# ⚠️ Disclaimer & Educational Notice

**This project is created strictly for educational, research, and hobbyist purposes to demonstrate the capabilities of real-time object detection using Python and YOLOv8.** * **No Game Modification:** This software does not inject code, alter game memory, or modify any proprietary game files. It operates entirely as an external computer vision system analyzing local screen pixels.
* **Terms of Service Warning:** Utilizing automated tools, macros, or assistive vision scripts in online multiplayer environments violates the Terms of Service (ToS) of most game developers and can result in permanent account bans. 
* **Intended Use:** This repository is intended to be run exclusively in private offline testing environments, single-player modes, or custom training maps where automated tools are explicitly allowed. 
* **Liability:** The author assumes no responsibility or liability for any account bans, hardware flags, or misuse of this code by third parties. Use responsibly and at your own risk.

# AI-Aimbot-with-python-and-YOLO-v8
To run this you need Python 3.12.9

1. Go to code and download the ZIP file.
2. Copy the ZIP to Desktop.
3. Extract the ZIP.
4. Open the requirements .txt files. If you are using Nvidia, only download Nvidia requirements. If not, download the non-Nvidia requirements only. To download the requirements, open CMD, copy and paste one and wait. After it finishes, copy the next and install it. You will need to install all of them to run the aimbot.
5. Open `main.py` after installation of the requirements and the aimbot will start running and a GUI will pop up.

**Note:** Don't change any name or location or remove any files from the folder `main.py` is found, otherwise it's not going to work.

# Recommended settings and how to use 
1. Select weight file. I would recommend the universal one to use. Select it.
2. Press on enable operational tracker, then select what button you want to enable auto aiming. You need to be pressing on the button for the aiming to work. If you take your finger off the selected button, it won't track any player. You need to be pressing on the selected button to auto aim for you.
3. Select the input driver. I would recommend **pywin32**.
4. I would recommend not changing the next group of settings except on the target axis alignment. It's responsible for the place it auto aims. Values like `0.19` aim at the player's head and values like `90` aim at the player's feet.
5. The next setting is active canvas FOV. Higher values make the circle go bigger, which is more detection range. The script detects what is in the circle only and aims on what is in the circle only. I don't recommend going more than 500px.
6. The next setting I would recommend keeping it on 40 to 45. It's the confidence rate. Like if it was not confident, it won't aim at the target, but putting confidence too low can make it aim at walls and random stuff.
7. Lastly, choose a color for the circle on the screen. I would recommend blue.

# Other Information 
This is going to support the GPU for Nvidia users only. This doesn't support AMD. For AMD, it's going to use the CPU instead of GPU, and also for integrated graphics users. 

The CPU only will make the aimbot run on lower FPS. 

If it doesn't work, try using VSCode venv. 

The requirements is going to need about 7 GB and take time for installation.

This script supports all `.pt` or `.onnx` files. Just drop them on the same folder or place as `main.py`.
