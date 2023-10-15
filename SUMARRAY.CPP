

#include<iostream.h>
#include<conio.h>
class array
{
 int a[25],n;
 public:
	int elements();
	void readarray();
	void disparray();
	void addarray(array,array);
 };
 int array:: elements()
 {
   cin>>n;
   return n;
 }
 void array::readarray()
 {
   for(int i=0;i<n;i++)
      cin>>a[i];
 }
 void array::disparray()
 {

   for(int i=0;i<n;i++)
     cout<<a[i]<<"\n" ;
 }
 void array::addarray(array x,array y)
 {
   n=x.n;
   for(int i=0;i<n;i++)
     a[i]=x.a[i]+y.a[i];
 }
 void main()
 {
   clrscr();
   int m,n;
   array obj1,obj2,obj3;
   cout<<"enter the elements in first array\n";
   m=obj1.elements();
   cout<<"enter the elements in second array\n";
   n=obj2.elements();
   if(m==n)
   {  cout<<"enter the first array";
      obj1.readarray();
      cout<<"enter the second array";
      obj2.readarray();
      obj3.addarray(obj1,obj2);
      obj3.disparray();
   }
   else
      cout<<"addition can't performed" ;


 getch();
 }




