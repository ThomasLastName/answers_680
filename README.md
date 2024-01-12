# MATH 680 ANSWERS TO EXERCISES IN LABS
These are the answers to the exercises presented in labs for the class MATH 680 at Texas A&M university, which can be found in my repo [labs_680](https://github.com/ThomasLastName/labs_680). I wrote these as the TA for the class in spring 2024.

---

# Usage
Each of the files in my [labs_680](https://github.com/ThomasLastName/labs_680) repo contains multiple reproducible demonstations, as well as 1-3 exercises. Contained within _this_ repo are my solutions to those exercises.


---

# Prerequisites for Using This Code
Besides some standard libraries, this repo depends on the folder of code [quality_of_life](https://github.com/ThomasLastName/quality_of_life), which is simply a collection of coding shortcuts that I want to be able to use when writing the demos for this class, and in every other python project that I engage in.

**List of Requirements in Order to Use this Code:**
- [x] Have python installed and know how to edit and run python files
- [x] **(important)** Know the directory of your python's `Lib` folder (see below)
- [x] Have the repository [quality_of_life](https://github.com/ThomasLastName/quality_of_life) already stored in your `Lib` folder. This has its own installation steps, similar to the steps for this repo. See its REDAME for more info.
- [x] Have the prerequisite standard packages installed: 
    - `numpy` and `matplotlib` for minimal functionality
    - `tensorflow`, `pytorch`, `sklearn` for complete functionality

**More on the Directory of Your Python's `Lib` Folder:** Unless you made a point of moving python after installing it, this will be the directory to which you installed python, plus `\Lib`. For example, on my personal computer, python is located in the folder  `C:\Users\thoma\AppData\Local\Programs\Python\Python310`, within which many things are contained, including a folder called `Lib`. Thus, the directory of my `Lib` folder is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib`. For reference, this is also where many of python's base modules are stored, such as `warnings.py`, `pickle.py`, and `turtle.py`.

I recommend having the directory where your version python is installed written down somewhere. If you do not know this location, I believe you can retrieve it in the interactive python terminal by commanding `import os; import sys; print(os.path.dirname(sys.executable))`. Thus, in Windows, you can probably just open the command line and paste into it `python -c "import os; import sys; print(os.path.dirname(sys.executable))"`. That said, if you have multiple versions of python on you'll computer, then you may want to be mindful of which version's terminal you're executing `import os; import sys; print(os.path.dirname(sys.executable))` in.

---

# Installation

Basically, just create a folder called `answers_680` inside of your python's `Lib` folder, and fill it with the files from this repository.

---

## Detailed Installation Instructions Using git (recommended)


**Additional Prerequisites Using git:**
- [x] Have git installed on your computer

**Installation Steps Using git:**
Navigate  to the `Lib` folder of the version of python you want to use. Once there, command `git clone https://github.com/ThomasLastName/answers_680.git`, which will create and populate a folder called `answers_680` in the same directory.

For example, given that the directory of my `Lib` folder is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` on my personal computer, I would navigate there by pasting `cd C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib` into the Windows command line, and then I would paste `git clone https://github.com/ThomasLastName/answers_680.git`.

**Subsequent Updates Using git:**
Navigate to the directory of the folder that you created, and within that directory command `git pull https://github.com/ThomasLastName/answers_680.git`.

For instance, to continue the example above, if I created the folder `answers_680` in my `Lib` folder `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib`, then the directory of the folder `answers_680` is `C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib\answers_680`. I'll want to navigate there in the Windows command line by pasting `cd C:\Users\thoma\AppData\Local\Programs\Python\Python310\Lib\answers_680` and, then, I'm ready to paste `git pull https://github.com/ThomasLastName/answers_680.git`.

---

## Detailed Installation Instructions Using the Graphical Interface

**Installation Steps Using the Graphical Interface:**
Click the colorful `<> Code` button at [https://github.com/ThomasLastName/answers_680](https://github.com/ThomasLastName/answers_680) and select `Download ZIP` from the dropdown menu. This should download a zipped folder called `answers_680` containing within it an unzipped folder of the same name, which you just need to click and drag (or copy and paste) into the `Lib` folder of your preferred version of python.

**Subsequent Updates Using the Graphical Interface:**
You'll have to repeat the process, again. When you attempt to click and drag (or copy and paste) the next time, your operating system probably prompts you with something like "These files already exist! Are you tweaking or did you want to replace them?" and you can just click "replace" or whatever it prompts you with.

