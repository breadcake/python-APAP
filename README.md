# python-APAP
Simple implementation of APAP using opencv-python

*As-Projective-As-Possible Image Stitching with Moving DLT* 这篇论文的Python实现，仅实现了两幅拼接
```
opencv-python==4.4.0.42
numpy
scipy
```

运行结果：
<table>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/bc637bc40e344964913a3541c4d86a21.png?x-oss-process" >linear_hom</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/ebe6917af76e4cc29c2af05e3fbd08fd.png?x-oss-process"  >linear_mdlt</center></td>
    </tr>
    <tr>
        <td ><center><img src="https://img-blog.csdnimg.cn/b115ad9326fc44e18278c9adb81dc9de.png?x-oss-process" >linear_hom</center></td>
        <td ><center><img src="https://img-blog.csdnimg.cn/2d973ce4d45646e2951e842fd2af560d.png?x-oss-process"  >linear_mdlt</center></td>
    </tr>
</table>
