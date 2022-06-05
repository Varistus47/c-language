#include <stdio.h>
#include <conio.h>

int main(){
    int n,a[100],i;
    void sortarray(int*,int);
    printf("\nenterno of elements in array:");
    scanf("%d",&n);
    printf("\nenter array elements:");
    for(i=0;i<n;i++){
        scanf("%d",&a[i]);
    }
    sortarray(a,n);
    printf("\nafter sorting\n");
    for(i=0;i<n;i++){
        printf("%d\n",a[i]);
    }

}
void sortarray(int* arr,int num){
    int i,j,temp;
    for(i=0;i<num;i++){
        for(j=i+1;j<num;j++){
            if(arr[i]>arr[j]){
                temp=arr[i];
                arr[i]=arr[j];
                arr[j]=temp;
            }
        }
    }
}