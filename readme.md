>背景
>>需要写一个解析器， 来实现如下功能
>>>实现基本的替换功能  
>>>> a = x y z  
>>>> a = <ref> x  
>>>> a = x | z | y  
>>>> a = [x y z] x y  
>>>> a = [x | y | z] x y  
>>>> b = <a> x y
>>>> a = magic(b|c|e, var, norm_value)
>>>> a = '2232'