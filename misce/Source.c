#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<Windows.h>
//#include<unistd.h>
#include<stdbool.h>
//#include<dirent.h>
#include<tchar.h>
#include <strsafe.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<stdarg.h>


//=========================HASH TABLES =======================================================================
typedef struct mydata_tag{
	int used;
	int key;
	char name[30];
}mydata;

int hash_key(char *name)
{
	int key, len, k;
	key = 0;
	len = strlen(name);
	for (k = 0; k < len; k++)
	{
		key += name[k];
	}
	key %= 17;
	return key;
}

void init_table(char *filename, int size)
{
	FILE *fp;
	mydata data;
	int k;
		memset(&data, 0, sizeof(mydata));
		fp = fopen(filename, "w+");
		if (fp == NULL)
		{
			perror("fopen:init_table\n");
			exit(1);
		}
		for (k = 0; k < size; k += 1)
		{
			fwrite(&data, sizeof(mydata), 1, fp);
		}
		fclose(fp);
}

void insert_data(int key, char *name, char *filename)
{
	FILE *fp;
	mydata data,slot;
	int pos;

	pos = key;

	data.used = 1;
	data.key = key;
	strcpy(data.name, name);

	fp = fopen(filename, "r+");
	if (fp == NULL)
	{
		perror("fopen:insert_data\n");
		exit(1);
	}

	while (1)
	{
		fseek(fp, pos*sizeof(mydata), SEEK_SET);
		fread(&slot, sizeof(mydata), 1, fp);
		if (slot.used != 1)
		{
			break;
		}
		printf("Collision\n");
		
		pos++;
		pos %= 17;
	}
	printf("pos=%d\n", pos);
	fseek(fp, pos*sizeof(mydata), SEEK_SET);
	fwrite(&slot, sizeof(mydata), 1, fp);
	fclose(fp);
}

void print_buckets(char *filename)
{
	FILE *fp;
	mydata data;
	int k;

	fp = fopen(filename, "r+");
	if (fp == NULL)
	{
		perror("fopen:print_buckets\n");
		exit(1);
	}
	for (k = 0; k < 17; k++)
	{
		fseek(fp, sizeof(mydata), SEEK_SET);
		fread(&data, sizeof(mydata), 1, fp);
		printf("used=%d,key=%d,name=%s\n", data.used, data.key, data.name);
	}
	fclose(fp);

}
//=========================BINARY TREE ========================================================================

typedef struct mynode_tag{
	int data;
	struct mynode_tag *left;
	struct mynode_tag *right;
}mynode;

void insert(mynode **rt, int num)
{
	mynode *tmp;
	if (*rt == NULL)
	{
		 tmp = (mynode *)malloc(sizeof(mynode));
		 if (tmp == NULL)
		 {
			 fprintf(stderr, "Unable to allocate mynode\n");
			 exit(1);
		 }
		 tmp->data = num;
		 *rt = tmp;
	 }
    else
    {
	    if (num > (*rt)->data)
	    {
		    insert(&(*rt)->right, num);
	    }
        else
        {
	         insert(&(*rt)->left, num);
        }
	
   }
	

}

void print_nodes(mynode *rt)
{
	if (rt == NULL)
		return;
	if (rt->left != NULL)
		print_nodes(rt->left);
	    printf("%d\n", rt->data);
		if (rt->right != NULL)
			print_nodes(rt->right);

}

//=====================DOUBLE POINTER ==============================================================

void nameSwap(char *((**before)[3]),char *((**after)[3]))
{
	char *((*tmp)[3]);

	tmp = *before;
	*before = *after;
	*after = tmp;


}
//========================FSEEK =====================================================================
int seekAndReplace()
{
	char name[256]; // = "Terrence Kavyu Muthoka";
	char dataName[256];
	strcpy(name, "C:\\Users\\y9ck3\\Documents\\Visual Studio 2013\\Projects\\ConvertedBase\\NA_all");
	printf("%s\n", name);
	strcat(name, "\\\\");
	printf("%s\n", name);



	strcpy(dataName, "RETAILERS");
	strcat(dataName, ".csv");

	strcat(name, dataName);
	printf("%s\n", name);


	FILE *myFile;
	myFile = fopen(name, "a");
	if (myFile != NULL)
	{
		perror("Couldn't Find the File\n");
		exit(1);
	}

	fprintf(myFile, "%s %s %s %d\n", "we", "are", "in", 2013);
	fclose(myFile);

	//printf("%s\n", dataName);


	//strcat(name, dataName);
	//printf("%s\n", name);

	//fseek(name, 0, SEEK_END);
	//printf("%s\n", name);

	//fputs("\\*", name);
	//printf("%s\n", name);

	return 0;
}
//====================MULTIDIMENSIONAL ARRAYS ========================================================
int matrixProduct()
{
	int matrix1[2][3];
	int matrix2[3][3];
	int product[2][3];

	matrix1[0][0] = 1; matrix1[0][1] = 2; matrix1[0][2] = 3; matrix1[1][0] = 5; matrix1[1][1] = 1; matrix1[1][2] = 1;
	matrix2[0][0] = 1; matrix2[0][1] = 0; matrix2[0][2] = 1; matrix2[1][0] = 1; matrix2[1][1] = 1; matrix2[1][2] = 0;
	matrix2[2][0] = 1; matrix2[2][1] = 1; matrix2[2][2] = 1;

	int l=0, k=0,j=0;
	product[l][k] = 0;
	for (l = 0; l < 2; l++)
	{
		for (k = 0; k < 3; k++)
		{
			for (j = 0; j < 3; j++)
			{
				product[l][k] = product[l][k] + matrix1[l][1] * matrix2[1][k];
			}
			

		}
	}
	int m, n;
	for (m = 0; m < 2; m++)
	{
		for (n = 0; n < 3; n++)
		{
			
			printf("%d\t\t",product[m][n]);// = product[l][k] + matrix1[l][1] * matrix2[1][k];

		}

		printf("\n");
	}

	return 4;
}

//==============================ABSOLUTE VALUE =======================================================
double abso(double y)
{
	double a;
	if (y < 0)
	{
		a = -y;
	}
	else{
		a = y;
	}

	return a;
}

//=============================SQUARE ROOT ==========================================================

double squareRT(double number)
{
	/*
	double x = number/2;
	double y;
	while (abso(power(x, 2) - number)>0.1)
	{
		y = (power(x, 2) + number) / (2 * x);
		x = y;
	}

	printf("%f", x);
	*/
	return number;
	
}

//====================PRIMES =========================================================================
int primeBefore(int x)
{
	int i, j;
	bool isPrime=true;
	int k=0;
	//int primes[x];
	for (i = 2; i < x; i++)
	{
		for (j = 2; j < i; j++)
		{
			if ((i%j)==0)
			{
				isPrime=false;
			}
            else
            {
				//isPrime = true;
				//k++;
				//printf("%d\n", i);
            }
			if (isPrime = true);
			printf("%d\n", i);
		}
		
	}
	//int primes[k];

	return k;

}

//==================================INTEGRATION =======================================================

long double  integrate(int a, int b)
{
	long double error = 0.0001;
	long double sum = 0;
	int n, k;
	long double s;
	//while (sum != 2.33)
	//{
		n = (int)((b - a) / error);
		s = 0;
		for (k = 0; k < n; k++)
		{
			s = s + (error)*(a + k*error + (error / 2))*(a + k*error + (error / 2));
		}

		//error = error / 10;
		sum = s;
	//}

	printf("%lf\n", sum);
	return sum;


}

//========================POWER FUNCTION ===============================================================
int power(int base, int exp)
{
	int i;
	int result=1;
	for (i = 0; i < exp; i++)
	{
		result = result*base;
	}

	//printf("%d\n", result);
	return result;

}
//==================TOLOWER ===============================================================================
int toLower( char myWord[]){
	char smallLetters[] = "abcdefghijklmnopqrstuvwxyz";
	char bigLetters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	//printf("%d %d\n", strlen(smallLetters), strlen(bigLetters));
	//char myWord[] = "CompANIESAND QUANTITATIVE RISK ANLYSTS";

	//printf("%s\n", myWord);

	unsigned int k, l;
	for (k = 0; k < strlen(myWord); k++)
	{
		for (l = 0; l < strlen(bigLetters); l++)
		{
			//(myWord[k] = bigLetters[l]) ? smallLetters[l] : myWord[k];
			//break;
			if (myWord[k] == bigLetters[l])
			{
				myWord[k] = smallLetters[l];
				break;
			}
			else{
				continue;
			}

		}

	}

	printf("%s\n", myWord);

	return strlen(myWord);
}

//==================================BASE B TO BASE 10 =================================================================
int conv(char number[], int fromBase)
{
	char allValues[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	unsigned int k, l;
	long long int result = 0;
	unsigned int len = strlen(number);

	for (k = 0; k < len; k++)
	{
		for (l = 0; l < strlen(allValues); l++)
		{
			if (number[k]==allValues[l])
			{
				int pow = power(fromBase,((len - 1) - k));
				result = result +  l*pow;
				break;
			}
            else
            {
				continue;

            }
		}
	}

	printf("In base 10 is %ld\n", result);

	return 5;

}

//==================DIRECTORY LIST OF FILES ============================================================================

void GetNumberOfFilesInDir(char DatasetName[], char WorkingDirectory[])
{
	
	int total_files = 0;
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	//LPCWSTR lpath = _T("C:\\Users\\y9ck3\\Documents\\Visual Studio 2013\\Projects\\ConvertedBase\\NA_all\\*");
	//LPCWSTR lpath = strcat(WorkingDirectory, "\\\\*");
	char *FilePath = strcat(WorkingDirectory, "\\\\*") ;
	LPCWSTR lpath = (LPCWSTR)FilePath;

	hFind = FindFirstFile((LPCWSTR)lpath, &FindFileData);

	if (hFind == INVALID_HANDLE_VALUE)
	{
		perror("File not found \n");
		exit(1);
	}
	else{
		FILE *ptr_readFile;
		FILE *ptr_writeFile;
		char line[140];
		int fileCounter = 1, lineCounter = 1;
		while ((FindNextFile(hFind, &FindFileData)) != 0)
		{

			total_files++;

			ptr_readFile = fopen((const char *)FindFileData.cFileName, "r");
			if (ptr_readFile != NULL)
			{
				perror("Read File could not be found\n");
				exit(1);
			}

			strcat(WorkingDirectory, "\\\\");
			strcat(DatasetName, ".csv");

			ptr_writeFile = fopen(strcat(WorkingDirectory, DatasetName), "w");
			if (ptr_writeFile != NULL)
			{
				perror("Write File could not be found\n");
				exit(1);
			}
			while (fgets(line, 140, ptr_readFile) != NULL)
			{
				fprintf(ptr_writeFile, "%s", line);
			}
			fclose(ptr_readFile);
			fclose(ptr_writeFile);
		}


	}
	total_files--;
	FindClose(hFind);
	


	
}


//=======================QUEUEING ==================================================================
#define QUEUESIZE 10
int queue[QUEUESIZE], f = 0, r = -1;

// Check if queue is full
int queuefull() {
	if (r == QUEUESIZE - 1) {
		return 1;
	}
	return 0;
}

// Check if the queue is empty
int queueempty() {
	if (f > r) {
		return 1;
	}
	return 0;
}

// Show queue content
int queueshow() {
	int i;
	if (queueempty()) {
		printf(" \n The queue is empty\n");
	}
	else {
		printf("Start->");
		for (i = f; i <= r; i++) {
			printf("%d ", queue[i]);
		}
		printf("<-End");
	}
	return 0;
}

// Perform an insert operation.
int queueinsert(int oneelement) {
	if (queuefull()) {
		printf("\n\n Overflow!!!!\n\n");
	}
	else {
		++r;
		queue[r] = oneelement;
	}
	return 0;
}

// Perform a delete operation
int queuedelete() {
	int elem;
	if (queueempty()) {
		printf(" \n The queue is empty\n");
		return(-1);
	}
	else {
		elem = queue[f];
		f = f + 1;
		return(elem);
	}
}


//=======================ISGOODVALUE()===================================================================

enum CellValue { NotErrorOrFormula,
	             Companies, 
				 Equities, 
				 Ratings };

bool IsGoodValue(char TestValue[], enum CellValue TestType)
{
	
	bool ReturnValue = false;
	
	char TestValue2[4];
	//memcpy(TestValue2,TestValue , 1);

	if (TestType == Companies){
		

		if (TestValue != "" &&  memcpy(TestValue2, TestValue, 1) != "#" &&  memcpy(TestValue2, TestValue, 1) != "("
			&&  memcpy(TestValue2, TestValue, 1) != "=")
		{
			ReturnValue = true;
		}
	}
	else if (TestType == Equities){
		if (TestValue != "" &&  memcpy(TestValue2, TestValue, 1) != "#" &&  memcpy(TestValue2, TestValue, 1) != "("
			&&  memcpy(TestValue2, TestValue, 1) != "=")
		{
			ReturnValue = true;
		}
	
	}
	else if (TestType == NotErrorOrFormula){
	
		if (!(TestValue == "#ERROR" || TestValue == "#REFRESH" || TestValue == "#CIQINACTIVE"))
		{
			ReturnValue = true;
		}
	}
	else if (TestType == Ratings){
		if (TestValue == "")
		{
			ReturnValue = true;
		}
		else if (memcpy(TestValue2, TestValue, 1) != "#" &&  memcpy(TestValue2, TestValue, 1) != "("
			&&  memcpy(TestValue2, TestValue, 1) != "=")
		     {
			ReturnValue = true;
		     }
			
	}
	
	return ReturnValue;
	
}

//=================================BASE CONVERSION==================================================================

void convert(unsigned int nbase)
{
	//long long Hex= 115792089237316000000000000000000000000000000000000000000000000000000000000000.00;


	 unsigned int  num;
	unsigned int  r;
	//long long number = 115792089237316000000000000000000000000000000000000000000000000000000000000000.00;
	num = 63047;
	//nbase = 5;
	char allValues[] = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
	char converted[33];
	char *p;
	p = &converted[32];
	*p = '\0';

	while (num !=0)
	{
		r = num % nbase;
		p = p - 1;
		*p= allValues[r];

		num =num/nbase;
	}

	printf(" in base %ld is %s\n",nbase, p);

	}
//=====================================================MAIN================================================================
int main(int argc, char *argv[])

{
	// ========================================BASE CONVERSION==============================================

	//double  Hex = 115792089237316000000000000000000000000000000000000000000000000000000000000000.00;

	//printf("%f", Hex);

	/*convert(2);
	convert(3);
	convert(4);
	convert(5);
	convert(6);
	convert(7);
	convert(8);
	convert(9);
	convert(10);
	convert(11);
	convert(12);
	convert(13);
	convert(14);
	convert(15);
	convert(16);
	convert(17);
	convert(18); convert(19); convert(20); convert(21); convert(22); convert(23); convert(24); convert(25); convert(26);
	convert(27); convert(28); convert(29); convert(30); convert(31); convert(32); convert(33); convert(34); convert(35);
	convert(36); convert(37); convert(38); convert(39); convert(40); convert(41); convert(42); convert(43); convert(44);
	*/
	//convert(36);

	// =========================CREATING LOG FILE =================================================================================
	/*
	time_t currenttime;
	char *timestring;
	//char buffer[40];

	currenttime = time(NULL);
	timestring = ctime(&currenttime);

	FILE *logFile;
	//errno_t err;

	logFile = fopen("C:\\Users\\y9ck3\\Documents\\Visual Studio 2013\\Projects\\ConvertedBase\\logFile.log", "w");
	if (logFile == NULL)
	{
	perror("File could not be found\n");
	exit(1);
	}
	fprintf(logFile, "Process begun at                 %s\n", timestring);

	//fwrite("Process begun at", sizeof(char), 16, logFile);
	//fprintf(logFile, "\nProcess begun at");

	int TestRows, ThisTestRowsMaxCount, MaxRowsToTest, MaxRowsToExcel; // , TotalTestRowsMaxCount;
	MaxRowsToTest = 30000000;
	MaxRowsToExcel = 300000;
	ThisTestRowsMaxCount = 70000;
	if (ThisTestRowsMaxCount > MaxRowsToExcel)
	{
	TestRows = MaxRowsToExcel;
	}
	else
	{
	TestRows = ThisTestRowsMaxCount;
	}


	char DatasetName[] = "AMR_Retailers";
	char  DataType[] = "Companies";
	int Identifiers = 75;
	int Variables = 14;
	char WorkingPath[] = "ConvertedBase\\logFile.log";
	int MAX_REFRESH_COUNT = 11;
	int MAX_APP_LOAD_COUNT = 30;

	//fwrite("TestRows", sizeof(int), 16, logFile);
	fprintf(logFile, "========================== CIQ Download Tool ===================================\n");
	fprintf(logFile, "Process begun at                 %s\n", timestring);
	fprintf(logFile, "Dataset Name:                    %s\n", DatasetName);
	fprintf(logFile, "Download Type:                   %s\n", DataType);
	fprintf(logFile, "Identifiers:                     %d\n", Identifiers);
	fprintf(logFile, "Variables:                       %d\n", Variables);
	fprintf(logFile, "\n");
	fprintf(logFile, "Working Directory:               %s\n", WorkingPath);
	fprintf(logFile, "Maximum rows sent to Excel:      %d\n", MaxRowsToExcel);
	fprintf(logFile, "Maximum rows test at a time:     %d\n", MaxRowsToTest);
	fprintf(logFile, "Maximum Excel refresh count:     %d\n", MAX_REFRESH_COUNT);
	fprintf(logFile, "Maximum Excel reload count:      %d\n", MAX_APP_LOAD_COUNT);
	fprintf(logFile, "=============================== Begin ===========================================");
	fprintf(logFile, "\n");
	fprintf(logFile, "TestRows :                    %d\n", TestRows);
	fprintf(logFile, "Begin Download at             %s\n", timestring);

	int k;
	for (k = 0; k < 10; k++)
	{
	fprintf(logFile, "Count:                       %d\n", k);
	//printf("Count: %d\n", k);

	Sleep(5000);
	}
	currenttime = time(NULL);
	timestring = ctime(&currenttime);
	fprintf(logFile, "Count ended at             %s\n", timestring);
	*/


	//=====================TIME IN UNIX ==========================================================================================
	/*
		time_t currenttime;
		char *timestring;
		char buffer[40];

		currenttime = time(NULL);
		timestring = ctime_s(buffer,sizeof buffer,&currenttime);

		printf("The time is %s\n", buffer);
		printf("Seconds elapsed since Jan 1 1970 are %ld\n", currenttime);
		printf("Hours elapsed since Jan 1 1970 are %ld\n", currenttime/3600);
		printf("Days elapsed since Jan 1 1970 are %ld\n", currenttime/86400);
		printf("Years elapsed since Jan 1 1970 are %ld\n", currenttime/31557600);

		//printf("The time is %s\n", *timestring);
		//printf("The time is %s\n", ctime(&currenttime));



		struct tm str_time;
		time_t time_of_day;

		str_time.tm_year = 1980 - 1900;
		str_time.tm_mon = 8;
		str_time.tm_mday = 27;
		str_time.tm_hour = 12;
		str_time.tm_min = 3;
		str_time.tm_sec = 5;
		str_time.tm_isdst = 0;

		time_of_day = mktime(&str_time);
		timestring = ctime_s(buffer, sizeof buffer, &time_of_day);
		printf("\nnew Time : %s\n",buffer);
		//printf("new Time : %s\n", ctime_s(buffer, sizeof buffer, &time_of_day));

		*/
	//================================HOW TO USE TIME IN WINDOWS ======================================
	/*
		//typedef struct _SYSTEMTIME {
		//WORD wYear;
		//WORD wMonth;
		//WORD wDayOfWeek;
		//WORD wDay;
		//WORD wHour;
		//WORD wMinute;
		//WORD wSecond;
		//WORD wMilliseconds;
		//} SYSTEMTIME;

		SYSTEMTIME myTime;
		GetSystemTime(&myTime);
		printf("Year:%d\n Month:%d\n Date:%d\n Hour:%d\n Min:%d\n Second:%d\n", myTime.wYear, myTime.wMonth,
		myTime.wDay, myTime.wHour, myTime.wMinute, myTime.wSecond);
		*/

	//==================================ISGOODVALUE ===========================================================
/*
	char name[40] = "ERROR"; // "#AMR_Retailers"
	bool myValue = IsGoodValue(name, Companies);
	printf("%d\n", myValue);
	*/
	//bool myValue = IsGoodValue(name, Equities);
	//printf("%d\n", myValue);

	//bool myValue = IsGoodValue(name, NotErrorOrFormula);
	//printf("%d\n", myValue);

	//bool myValue = IsGoodValue(name, Ratings);
	//printf("%d\n", myValue);
	

	//=============================QUEUEING ==========================================================
/*
	int option, element;
	char    ch;

	do {
		printf("\n Press 1-Insert, 2-Delete, 3-Show, 4-Exit\n");
		printf("\n Your selection? ");
		//scanf("%d", &option);
		scanf_s("%d", &option, 1);
		switch (option) {
		case 1:
			printf("\n\nContent to be Inserted?");
			//scanf("%d", &element);
			scanf_s("%d", &element, 1);
			queueinsert(element);
			break;
		case 2:
			element = queuedelete();
			if (element != -1) {
				printf("\n\nDeleted element (with content %d) \n", element);
			}
			break;
		case 3:
			printf("\n\nStatus of the queue\n\n");
			queueshow();
			break;
		case 4:
			printf("\n\n Ending the program \n\n");
			break;
		default:
			printf("\n\nInvalid option, please retry! \n\n");
			break;
		}
	} while (option != 4);
	*/

	//==========================MERGING FILES ===========================================================================


	FILE *ptr_readFile;
	FILE *ptr_writeFile;
	char line[140];
	//char fileOutPutName[40];
	int fileCounter=1, lineCounter = 1;
	errno_t err1;
	errno_t err2;

	err1 = fopen_s(&ptr_readFile, "J:\\Quantitative Risk Analytics\\User-TM\\NA_all\\NA_all_QtrlyEquityData_00001.csv", "r");
	if (err1 != 0)
	{
		perror("Read File could not be found\n");
		exit(1);
	}

	//sprintf(fileOutPutName, "file number:%d", fileCounter);
	//err2 = fopen_s(&ptr_readFile, "J:\\Quantitative Risk Analytics\\User-TM\\NA_all\\NA_all_QtrlyEquityData_00001.csv", "r");
	err2 = fopen_s(&ptr_writeFile, "J:\\Quantitative Risk Analytics\\User-TM\\NA_all\\fileOutPutName.csv", "w");
	if (err2 != 0)
	{
		perror("Write File could not be found\n");
		exit(1);
	}
	while (fgets(line,140,ptr_readFile) != NULL)
	{
		fprintf(ptr_writeFile, "%s", line);
	}
	fclose(ptr_readFile);


	FILE *ptr_readFile2;
	errno_t err3;
	err3 = fopen_s(&ptr_readFile2, "J:\\Quantitative Risk Analytics\\User-TM\\NA_all\\NA_all_QtrlyEquityData_00002.csv", "r");
	if (err3 != 0)
	{
		perror("Read File2 could not be found\n");
		exit(1);
	}
	while (fgets(line, 140, ptr_readFile2) != NULL)
	{
		fprintf(ptr_writeFile, "%s", line);
	}
	fclose(ptr_readFile2);



	FILE *ptr_readFile3;
	errno_t err4;
	err4 = fopen_s(&ptr_readFile3, "J:\\Quantitative Risk Analytics\\User-TM\\NA_all\\NA_all_QtrlyEquityData_00003.csv", "r");
	if (err4 != 0)
	{
		perror("Read File3 could not be found\n");
		exit(1);
	}
	while (fgets(line, 140, ptr_readFile3) != NULL)
	{
		fprintf(ptr_writeFile, "%s", line);
	}
	fclose(ptr_readFile3);

	fclose(ptr_writeFile);
	
	
	typedef struct _WIN32_FIND_DATA {
		DWORD    dwFileAttributes;
		FILETIME ftCreationTime;
		FILETIME ftLastAccessTime;
		FILETIME ftLastWriteTime;
		DWORD    nFileSizeHigh;
		DWORD    nFileSizeLow;
		DWORD    dwReserved0;
		DWORD    dwReserved1;
		TCHAR    cFileName[MAX_PATH];
		TCHAR    cAlternateFileName[14];
	} WIN32_FIND_DATA, *PWIN32_FIND_DATA, *LPWIN32_FIND_DATA;

GetNumberOfFilesInDir("C:\\Users\\y9ck3\\Documents\\Visual Studio 2013\\Projects\\ConvertedBase\\NA_all", "fileOutPutName2");


//==================TOLOWER ========================================================================================

/*
char words[] = "I LOVE PROGRAMMING IN C C++ C# etc";
toLower(words);
*/

//==================================BASE B TO BASE 10 =================================================================

/*
conv("SYOMBUA", 36);

*/
//================================POWER FUNCTION ==========================================================================
/*
power(2, 2);
power(2, 3);
power(2, 4);
power(2, 5);
*/
//==================================INTEGRATION =======================================================
/*
integrate(1,1000);

*/
//=============================SQUARE ROOT ==========================================================


//squareRT(2);



//====================PRIMES =========================================================================
/*
primeBefore(20);

*/

//====================MULTIDIMENSIONAL ARRAYS ========================================================
/*
matrixProduct();
*/

 //========================FSEEK =====================================================================
/*

seekAndReplace();
*/
//===================================POINTER ARITHMETIC ================================================

char *CIQ_GET_DATES_FORMULA_EQUITY[3] = { "=CIQ(A1,\"IQ_FIRSTPRICINGDATE\")", "=CIQ(A1,\"IQ_LASTPRICINGDATE\")", "=IFERROR(C1-B1 + 1,0)" };
char *CIQ_TIME_TEST_FORMULA_EQUITY[6] = { "=CIQ(A1,\"IQ_CLOSEPRICE\",B1,,,,\"USD\",\"H\")" };
char *TestFormula[3];
//strcpy(TestFormula, CIQ_TIME_TEST_FORMULA_EQUITY);
unsigned int k;
for (k = 0; k < sizeof(CIQ_GET_DATES_FORMULA_EQUITY); k += 1){
	strcpy((*TestFormula)[k], (*CIQ_GET_DATES_FORMULA_EQUITY)[k]);


}
unsigned int i;
for (i = 0; i < 3; i++)
{
	printf("%s ", (*TestFormula)[i]);

}

//=========DOUBLE POINTER =========================================================================================================
/*
int k;
char *before[3] = {"ONE","TWO","THREE"};
char *after[3] = { "MOJA", "MBILI", "TATU" };

char *((*ptr_before)[3]);
char *((*ptr_after)[3]);

ptr_before = &before;
ptr_after = &after;

for (k = 0; k < 3; k++)
{
	printf("%s\t%s\n", (*ptr_before)[k], (*ptr_after)[k]);
}

printf("\n");

nameSwap(&ptr_before, &ptr_after);

for (k = 0; k < 3; k++)
{
	printf("%s\t%s\n", (*ptr_before)[k], (*ptr_after)[k]);
}

printf("\n");

for (k = 0; k < 3; k++)
{
	printf("%s\t%s\n", (before[k]), (after[k]));
}
*/

//=========================BINARY TREE ========================================================================
/*
int k;
mynode *root=NULL;
int numbers[20] = {23,56,34,87,90,456,24,4,7,89,2,31,67,834,5,234,532,71,92,201};
for (k = 0; k < 20; k++)
{
	insert(&root, numbers[k]);
}

print_nodes(root);
*/
//====================HASH TABLES =================================================================================
/*
char *name[17] = { "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE",
"THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN" };
int k;
int key;
init_table("hashing", 17);
for (k = 0; k < 17; k++)
{
	key = hash_key(name[k]);
	insert_data(key, name[k], "hashing");
}

print_buckets("hashing");
*/
//===============FSYNC=======================================================================

/*
int fd, ret;
char *str = "This is my String.";

fd = open("MyFsync.txt","r+");
if (fd < 0)
{
	perror("fopen:MyFsync.txt");
	exit(1);
}

ret = write(fd, str, sizeof(str));
close(fd);
*/



	return 0;




}