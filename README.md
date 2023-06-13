# THNumber_img_classification ( without Deep Learning )
:pushpin: **Goal** :fire: : <br>
>การสอน computer ให้สามารถแยกแยะรูปภาพของเลขไทย ซึ่งเขียนด้วยลายมือ :crayon: ( ขนาด 28x28 pixels ) ว่าเป็นเลขอะไร <br>
>ด้วยการใช้เครื่องมือ *Machine Learning* ( โดยที่ **ไม่** มีการใช้ Neural Network หรือ Deep learning ในการ Train ) <br> 
>
>และสร้าง Application :toolbox::wrench: สำหรับคนที่ไม่สามารถเขียน Code ในการทำ Machine Learning (ML) ดังกล่าว ให้สามารถ Train Machine Learning ผ่าน App ได้ <br>

# <h3> Topics </h3>
สำหรับ Project นี้ เราจะแบ่งออกเป็น 2 หัวข้อ ได้แก่ <br>

>- [การทำ Machine Learning (ML)](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-machine-learning-ml)
>- การทำ Application :toolbox::wrench:

โดยจะเริ่มหัวข้อตามลำดับ <br>

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

  
  
