#include <stdio.h>
#include <conio.h>

void towerofhanoi(int n,char from,char to,char aux){
    if(n==1){
        printf("\n move disk 1 from peg %c to peg %c",from,to);
       return n;
    }

    towerofhanoi(n-1,from,aux,to);
    printf("\n Move disk %d from peg %c to peg %c",n,from,to);
    towerofhanoi(n-1,aux,to,from);

}
int main(){
    int n;
    printf("enter number of disks:");
    scanf("%d",&n);
    towerofhanoi(n,'A','C','B');
    return 0;
}