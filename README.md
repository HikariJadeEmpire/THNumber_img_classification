# THNumber_img_classification
**Goal** : <br>
>การสอน computer ให้สามารถแยกแยะรูปภาพของเลขไทย ซึ่งเขียนด้วยลายมือ ( ขนาด 28x28 pixels ) ว่าเป็นเลขอะไร <br>
>ด้วยการใช้เครื่องมือ Machine Learning ( โดยที่ **ไม่** มีการใช้ Neural Network หรือ Deep learning ในการ Train ) <br> 
>และสร้าง Application สำหรับคนที่ไม่สามารถเขียน Code ในการทำ Machine Learning (ML) ดังกล่าว ให้สามารถ Train Machine Learning ผ่าน App ได้ <br>

# <h3> Topics </h3>
สำหรับ Project นี้ เราจะแบ่งออกเป็น 2 หัวข้อ ได้แก่ <br>

>- การทำ Machine Learning (ML)
>- การทำ Application

โดยจะเริ่มหัวข้อตามลำดับ <br>

# <h3>การทำ Machine Learning (ML)</h3>
<h4>STEP 1</h4>
ในขั้นตอนเริ่มต้น เราก็จะทำการรวบรวม DATA : รูปภาพของตัวเลขไทยที่เขียนด้วยลายมือ ( ขนาด 28x28 pixels ) , เพื่อนำไป Train <br>
ซึ่งจะได้ออกมาเป็นตัวเลขละ 70 รูป โดยจะแสดงให้เห็นตัวอย่างของ DATA คร่าวๆ ดังนี้ : <br>
<br>

![output](https://github.com/HikariJadeEmpire/THNumber_img_classification/assets/118663358/42e4d3e4-8038-4e66-bc5d-846cf0556799)

จากนั้นเราจะทำการ Clean DATA ด้วยวิธีการ <br>
ตัดขอบภาพ >> แปลงเป็นภาพ ขาว-ดำ >> ทำการ Rescale ให้เป็น 28x28 pixels เหมือนตอนเริ่มต้น >> รวบรวม DATA แล้ว Transform ให้เป็น **.CSV** File
