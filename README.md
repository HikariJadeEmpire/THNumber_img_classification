# THNumber_img_classification ( without Deep Learning )
:pushpin: **Goal** :fire: : <br>
>การสอน computer ให้สามารถแยกแยะรูปภาพของเลขไทย ซึ่งเขียนด้วยลายมือ :crayon: ( ขนาด 28x28 pixels ) ว่าเป็นเลขอะไร <br>
>ด้วยการใช้เครื่องมือ *Machine Learning* ( โดยที่ **ไม่** มีการใช้ Neural Network หรือ Deep learning ในการ Train ) <br> 
>
>และสร้าง Application :toolbox::wrench: สำหรับคนที่ไม่สามารถเขียน Code ในการทำ Machine Learning (ML) ดังกล่าว ให้สามารถ Train Machine Learning ผ่าน App ได้ <br>

# <h3> Topics </h3>
สำหรับ Project นี้ เราจะแบ่งออกเป็น 2 หัวข้อ ได้แก่ <br>

>- [การทำ Machine Learning (ML)](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-machine-learning-ml)
>- [การทำ Application :toolbox::wrench:](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-application-toolboxwrench)

โดยจะเริ่มหัวข้อตามลำดับ <br>

# <h4>Languages & Tools</h4>

[![](https://img.shields.io/badge/code-python3.9-green?style=f?style=flat-square&logo=python&logoColor=white&color=2bbc8a)](https://www.python.org/)
[![](https://img.shields.io/badge/tools-jupyter-orange?style=f?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![](https://img.shields.io/badge/tools-dash-green?style=f?style=flat-square&logo=plotly&logoColor=white&color=2bbc8a)](https://dash.plotly.com/)
[![](https://img.shields.io/badge/tools-SkLearn-green?style=f?style=flat-square&logo=scikitlearn&logoColor=white&color=2bbc8a)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/tools-VSCode-blue?style=f?style=flat-square&logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/)
[![](https://img.shields.io/badge/tools-Pandas-green?style=f?style=flat-square&logo=pandas&logoColor=white&color=2bbc8a)](https://pandas.pydata.org/)
[![](https://img.shields.io/badge/tools-Numpy-green?style=f?style=flat-square&logo=numpy&logoColor=white&color=2bbc8a)](https://numpy.org/)
[![](https://img.shields.io/badge/OS-Mac-green?style=f?style=flat-square&logo=macos&logoColor=white)](https://www.apple.com/macos/ventura/)
[![](https://img.shields.io/badge/OS-Windows-green?style=f?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/)
[![](https://img.shields.io/badge/Git_Update-13_Jun_2023-brightgreen?style=f?style=flat-square&logo=github&logoColor=white)](https://github.com/)

# <h3>การทำ Machine Learning (ML)</h3>

**CODE** >>> <a href="https://colab.research.google.com/github/HikariJadeEmpire/THNumber_img_classification/blob/main/numberclassifier.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

<h4>STEP 1</h4>
ในขั้นตอนเริ่มต้น เราก็จะทำการรวบรวม DATA : รูปภาพของตัวเลขไทยที่เขียนด้วยลายมือ :crayon: ( ขนาด 28x28 pixels ) , เพื่อนำไป Train <br>
ซึ่งจะได้ออกมาเป็นตัวเลขละ 70 รูป โดยจะแสดงให้เห็นตัวอย่างของ DATA คร่าวๆ ดังนี้ : <br>
<br>

![output](https://github.com/HikariJadeEmpire/THNumber_img_classification/assets/118663358/42e4d3e4-8038-4e66-bc5d-846cf0556799)

จากนั้นเราจะทำการ Clean DATA :broom: ด้วยวิธีการ <br>
ตัดขอบภาพ >> แปลงเป็นภาพ ขาว-ดำ >> ทำการ Rescale ให้เป็น 28x28 pixels เหมือนตอนเริ่มต้น >> รวบรวม DATA แล้ว Transform ให้เป็น **.CSV** File <br>
  
**.CSV** File EXAMPLE :
  
![Capture](https://github.com/HikariJadeEmpire/THNumber_img_classification/assets/118663358/fd2af6c3-fbc8-4fa7-b6f0-7a3e211567b3)

# <h4>STEP 2</h4>
ขั้นตอนต่อจากนี้ เราจะทำการ **Cross Validation** ด้วยการใช้  [Pycaret :triangular_flag_on_post:](https://pycaret.gitbook.io/docs/) <br>
เพื่อค้นหา Model ที่มี Score โดยเฉลี่ยสูงที่สุด 3-5 อันดับแรก :trophy: แล้วนำไปปรับ ( Tune Model ) เพื่อนำไปใช้ในการ Train & Test ในขั้นตอนสุดท้าย <br>

*NOTE :* ลำดับของ Model อาจมีการเปลี่ยนแปลง เนื่องจากมีการ Re-sampling DATA ในทุกๆครั้งที่ Train
  
[Pycaret :triangular_flag_on_post:](https://pycaret.gitbook.io/docs/) score :
  
![cap0](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/aa3d9c75-a53b-4b92-9723-7f388194c5d9)

  # <h4>STEP 3</h4>
  Train & Test :books: <br>
  <br>
  
  ใน STEP นี้ เราจะนำ Model ที่มี Score เฉลี่ยดีที่สุด มาใช้ในการ Train & Test :books: โดยอาศัย Setting เดียวกันกับขั้นตอน **Cross Validation** <br>
  ซึ่ง Model ที่ได้มาก็คือ : **Extra Trees Classifier** :deciduous_tree:<br>
  
  ผลลัพธ์ที่ได้จากการ Test : **Extra Trees Classifier Model** :deciduous_tree: ได้ผลลัพธ์ออกมาดังนี้ : <br>
  <br>
  
  <img width="476" alt="Screenshot 2566-06-13 at 16 34 46" src="https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/fa1319d0-b7c5-4224-bf1c-3be210337f91">

  <br>
  <br>
  จาก Score ด้านบน จะพบว่าคะแนนที่ได้จากการ Test ค่อนข้างดีเยี่ยม โดยจะมีความแม่นยำอยู่ที่ราวๆ 90 % - 100 % <br>
  
  # <h3>การทำ Application :toolbox::wrench:</h3>
  
  
  
  
