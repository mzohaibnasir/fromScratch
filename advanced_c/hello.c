#include <stdio.h>

// int main(void){

//     char s[100];

//     printf("Enter the first string: ");
//     scanf("%s",s);
//     printf("Hello %s \n",s);
// }
// int main(void){
//     int x,y;
//     printf("Enter x: ");
//     scanf("%d", &x);

//     printf("Enter y: ");
//     scanf("%d", &y);
//     printf("Sum: %d\n", x+y);
// }

#include <stdio.h>
void meow(void);// function prototyoe
void meows(int n);// function prototyoe



int main(void)
{
    // int x, y;
    // int res_x, res_y;

    // printf("Enter x: ");
    // res_x = scanf("%d", &x);
    // if (res_x != 1)
    // {
    //     printf("Error reading input for x.\n");
    //     return 1;
    // }
    // else
    // {
    //     printf("res_x: %d\n", res_x);
    // }

    // printf("Enter y: ");
    // res_y = scanf("%d", &y);
    // if (res_y != 1)
    // {
    //     printf("Error reading input for y.\n");
    //     return 1;
    // }
    // else
    // {
    //     printf("res_y: %d\n", res_y);
    // }

    int x=3, y=5;

    printf("Sum: %d\n", x + y);
    if (x > y)
    {
        printf("x>y");
    } 
    else
    {
        if (x < y && x!=y)
        {

            printf("x<y");
        }
        else if (x == y)
        {
            printf("x==y\n");
        }
    }

    



    int counter =3;
    while (counter>0)
    {
        printf("\n%d",counter);
        /* code */
        counter--;
    }

    for(int i=0; i<3; i++){
        printf("\n%d",i);
    }
    meow();
    meows(5);   
}

// `x: 1 5 2; y:   ; sum:6 `The behavior you observed occurs because scanf reads the first integer (in this case, 1) and stops reading when it encounters a non-digit character, leaving the remaining characters (5 2) in the input buffer. When the program prompts for y, it immediately reads the next integer (5) from the input buffer without waiting for new input from the user.

// make hello
// ./hello

// \n quotation marks

void meow(void){
    printf("meow\n");
}

void meows(int n){
    for(int i=0; i<n;i++)
        printf("meow\n");
}


// # no function overloading