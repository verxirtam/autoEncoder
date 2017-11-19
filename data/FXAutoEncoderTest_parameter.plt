
set datafile separator ","
set pm3d map

#matlabのカラーパレット jet
#set palette defined ( 0 '#000090',1 '#000fff',2 '#0090ff',3 '#0fffee',4 '#90ff70',5 '#ffee00',6 '#ff7000',7 '#ee0000',8 '#7f0000')
#変化が細かいカラーマップ
set palette defined(0"#ffffff",0.8"#00008b",1.8"#2ca9e1",3"#008000",4.2"#ffff00",5"#eb6101",5.5"#8b0000")

min(x,y) = (x < y) ? x : y
max(x,y) = (x > y) ? x : y

splot [:][:][-2:2]'FXAutoEncoderTest_parameter.csv' matrix using 2:1:(max(min($3,2),-2)) with pm3d

pause -1
