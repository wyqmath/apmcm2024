%%%%%%%%%%%%异常值及缺失值检测%%%%%%%%%%%%
[num,txt]=xlsread("train.xlsx")  %num为数据中的数字部分，txt为数据中的文本部分
index=1;  %检测数据所在列
sales=num(2:end,index);  %提出所列数据并进行检测
rows=size(sales,1);  %对数据的行数（大小）读取
%缺失值检测 
nanvalue=find(isnan(sales));  %isnan函数来查找数组中的NaN值，该函数会返回一个逻辑数组，检查数据是否为缺失值,并返回缺失值的序数值
if isempty(nanvalue)  %isempty函数用来检查一个数组or变量是否为空，如果检测对象为空，该函数将返回逻辑值 true
    disp('没有缺失值')
else
    rows_=size(nanvalue,1);  %find函数的作用是检索数组中符合条件的元素的索引,这里用rows即表示了缺失值数量
    disp(['缺失值的个数为：',num2str(rows_),'缺失率为：',num2str(rows_/rows)]) 
end
%%%%%%%%%%%%Check_for_missing_values%%%%%%%%%%%%