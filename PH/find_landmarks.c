#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <popt.h>
#include <assert.h>
#include <omp.h>
#include <sys/time.h>
#include <sys/types.h>



/*===============================================
From CSCI5576 HW7
===============================================*/
double calctime(struct timeval start, struct timeval end) 
{
  double time = 0.0;
  time = end.tv_usec - start.tv_usec;
  time = time/1000000;
  time += end.tv_sec - start.tv_sec;

  return time;
}

/*===============================================
End from CSCI5576 HW7
===============================================*/

// Example of basic command line run after compilation:
//	./find_landmarks -i {witness input file} -o {file to output landmarks & distances} -l {number of landmarks} -w 0-100 
// for more information on parameters in the command line type ./find_landmarks --help



/*=============================================================
        Samantha Molnar
		Calculate euclidean distance matrix of witnesses then landmark selection.
		08/2017 Adding dimensionality option for witnesses
=============================================================*/
typedef enum { false, true } bool; // Provide C++ style 'bool' type in C
float           *witnesses;
float           *distances;
float           *euc_distance;
float           *times;
float           *velocities;
float           *norm_velocity;
float           *speeds;
float           *cov_matrices;
//timing
float            avg_time;
float            dev;
float            err;
float            sum;

//used to store calculated values
float            x,y,x1,yone,x2,y2,d,r,s,m,xi,xj,yi,yj; 
float            min_speed=1000000;
float            max_speed=-1000000;


//distance calculation options
float            use_hamiltonian = 1.0;
float            speed_amplify = 1.0;
float            orientation_amplify = 1.0;
float            ray_distance_amplify = 1.0;
float            straight_VB=0.0;
float            stretch = 1.0;
float            max_filtration_param = 0.0;
float            num_divs = 0.0;
float			 ham_param = 1.0;
float			 or_param = 1.0;

char            *file;
char            *wfile;
char            *wit;
char            *landmark_set;
FILE            *fp;

int64_t          num_wits=1;
int64_t          i,j,k,t,l;
int              m2_d = 0.0;
int              num_landmarks=1;
int              est;
int              start=0;
int              stop=1;
int              num_threads = 2;
int              d_cov = 0;


int              max_avg=50;
int              wit_pts; 
int             *landmarks;

bool             use_est=false;
bool             alt_wit=false;
bool             timing=false;

bool             use_euclidean = false;
bool             done = false;
bool 			 quiet = true;


void print_matrix(float *A);
int comp (const void * elem1, const void * elem2) ;
bool in_matrix(int* matrix,int size,int value);
int main(int argc, char* argv[]){
    char *parse;
//	struct timeval begin;
//	struct timeval end;
//    /***************** Command line argument parsing  *************************/
//    poptContext POPT_Context;  /* context for parsing command-line options */
//    char        POPT_Ret;      /* used for iterating over the arguments */

  struct poptOption optionsTable[] =
  { 
    { "input file",                'i', POPT_ARG_STRING,     &file,                         1, "Input file of witnesses",                                              0 },
    { "output file",               'o', POPT_ARG_STRING,     &wfile,                        2, "Where to output landmarks and their distances to each witness.",       0 },
    { "landmarks",                 'l', POPT_ARG_INT,        &num_landmarks,                3, "Number of landmarks to use.",                                          0 },
    { "witnesses",                 'w', POPT_ARG_STRING,     &parse,                        4, "Number of witnesses to use.  Can be an integer or a range of integers like x-y",      0 },
    { "evenly spaced in time",     'e', POPT_ARG_INT,        &est,                          5, "Use evenly spaced in time to select every x landmark.",                0 },
    { "time program",              't', POPT_ARG_NONE,       0,                             6, "Time each step of the code and output results.",                       0 },
    { "print everything",          'q', POPT_ARG_NONE,       0,                             7, "Print all output.  Use for debugging.",                                0 },
    { "set speed amplify",         'a', POPT_ARG_FLOAT,      &speed_amplify,                8, "Set the speed amplify variable.",                                      0 },
    { "set orientation amplify",   'y', POPT_ARG_FLOAT,      &orientation_amplify,          9, "Set the orientation amplify variable.",                                0 },
    { "set use hamiltonian",       'h', POPT_ARG_FLOAT,      &use_hamiltonian,             10, "Set the use hamiltonian variable.",                                    0 },
    { "set m2_d",                  'm', POPT_ARG_INT,        &m2_d,                        11, "Set the m2_d variable.",                                               0 },
    { "set ray dist amplify",      'r', POPT_ARG_FLOAT,      &speed_amplify,               12, "Set the ray distance amplify variable.",                               0 },
    { "number of threads",         'n', POPT_ARG_INT,        &num_threads,                 13, "Set the number of threads to use.",                                    0 },
    { "set straight_VB",           'v', POPT_ARG_FLOAT,      &straight_VB,                 14, "Set the straight_VB variable.",                                        0 },
    { "set stretch ",              's', POPT_ARG_FLOAT,      &stretch,                     15, "Set the stretch variable.",                                            0 },
    { "use euclidean ",            'c', POPT_ARG_NONE,       0,                            16, "Calculate distance using euclidean distance.",                         0 },
    { "cov",                       'x', POPT_ARG_INT,        &d_cov,                       17, "Calculate distance using covariance.",                                 0 },
    { "compute GI complex",        'f', POPT_ARG_FLOAT,      &max_filtration_param,        18, "Output edgelist for graph induced complex.",                           0 },
    { "number of divisions",       'd', POPT_ARG_FLOAT,      &num_divs,                    19, "Set number of divisions for GI complex.",                            0 },
    POPT_AUTOHELP
    { NULL, '\0', 0, NULL, 0}
  };
   // POPT_Context = poptGetContext(NULL, argc, (const char**) argv, optionsTable, 0);
   // poptSetOtherOptionHelp(POPT_Context, "[ Try --help for a more detailed description of the options]");
   // /* values are filled into the data structures by this function */
   //while ((POPT_Ret = poptGetNextOpt(POPT_Context)) >= 0
   //{
   //   switch (POPT_Ret)


    void eat_arg(int i, char* val)
    {
        switch (i)
        {
            case 1:
                file = val;
                if(!quiet)
                    printf("Input file:%s\n",file);
                break;

            case 2:
                wfile = val;
                if(!quiet)
                    printf("Output file:%s\n",wfile);
                break;

            case 3:
                num_landmarks = atoi(val);
                if(!quiet){
                    printf("Number of landmarks set to %d.\n",num_landmarks);
                    fflush(stdout);
                }
                break;

            case 4:
                parse = val;
                if(strchr(parse,'-')!=NULL){
                    parse=strtok(parse,"-");
                    i=0;
                    while(parse!=NULL){

                        if(i==0)
                            start = atoi(parse);
                        else
                            stop = atoi(parse);
                        i++;
                        parse=strtok(NULL,",-");
                    }
                }
                else{
                    start = 0;
                    stop = atoi(parse);
                }

                num_wits = stop-start;
                if(!quiet){
                    printf("Number of witnesses: %d\n",num_wits);
                    fflush(stdout);
                }
                break;

            case 5:
                est = atoi(val);
                use_est=true;
                if(!quiet)
                printf("Using evenly spaced in time to find landmarks.\n");
                break;

            case 6:
                if(!quiet)
                    printf("Timing selection processes.\n");
                timing = true;
                break;

            case 7:
                quiet = false;
                break;

            case 8:
                speed_amplify = atof(val);
                break;

            case 9:
                orientation_amplify = atof(val);
                break;

            case 10:
                use_hamiltonian = atof(val);
                break;

            case 11:
                m2_d = atoi(val);
                break;

            case 12:
                speed_amplify = atof(val);
                break;

            case 13:
                num_threads = atoi(val);
                printf("Number of threads: %d\n",num_threads);
                break;

            case 14:
                straight_VB = atof(val);
                break;

            case 15:
                stretch = atof(val);
                break;

            case 16:
                if(!quiet)
                    printf("Using euclidean distance.\n");
                use_euclidean = true;
                break;

            case 17:
                d_cov = atoi(val);
                break;

            case 18:
                max_filtration_param = atof(val);
                if(!quiet)
                    printf("Outputting edgelist for graph induced complex.\n");
                break;

            case 19:
                num_divs = atof(val);
                break;
        }
    }

    int NUMBER_ARGS = 19;    // number of all possible args
    int STRING_MAX_SIZE = 100;

    int switches [NUMBER_ARGS];
    char vals [NUMBER_ARGS][STRING_MAX_SIZE];

    FILE *sfile = fopen("find_landmark_arg_switches.txt", "r");
    FILE *vfile = fopen("find_landmark_arg_vals.txt", "r");

    int switch_;
    char value [STRING_MAX_SIZE];

    int i = 0;
    while (fscanf(sfile, "%d", &switch_) > 0)
    {
        switches[i] = switch_;
        i++;
    }


    i = 0;
    while (fscanf(vfile, "%s", &value) > 0)
    {
        strcpy(vals[i], value);
        i++;
    }


    fclose(sfile);
    fclose(vfile);


    for (i=0; i < NUMBER_ARGS; i++)
    {
        if (switches[i])
        {
            eat_arg(i + 1, vals[i]);
        }
    }

//  if (POPT_Ret < -1)
//  {
//    /* an error occurred during option processing */
//    fprintf(stderr, "%s: %s\n",
//    poptBadOption(POPT_Context, POPT_BADOPTION_NOALIAS),
//    poptStrerror(POPT_Ret));
//    return 1;
//  }
//  poptFreeContext(POPT_Context);

  /***************** End command line parsing *************************/



	
	/**************** Putting data into data structures ****************/
	fp=fopen(file,"r");
	if (fp == NULL) {
    	perror("Failed: ");
    	return 1;
	}
	printf("Reading in witnesses...");

	fflush(stdout);
	

	//reading in witnesses from file
	
	l = 0;
	i=0;
	int w=0;
	//need to change this to use getline
	char *pch;
	char *lpt=NULL;
	size_t sze=0;
	getline(&lpt,&sze,fp);
	pch = strtok(lpt," ");
	wit_pts=0;
	while (pch != NULL){
    	pch = strtok (NULL, " ");
    	wit_pts++;
    
    }
	printf("dimension of points=%d ...",wit_pts);
	rewind(fp); //set back to beginning

	witnesses         = (float*) calloc(num_wits*wit_pts,sizeof(float)); // x,y points for a witness
	distances         = (float*) calloc((float)num_wits*(float)num_wits,sizeof(float)); //distance between all witnesses
	euc_distance      = (float*) calloc((float)num_wits*(float)num_wits,sizeof(float)); //distance between all witnesses
	landmarks         = (int*)   calloc(num_landmarks,sizeof(int)); //landmark set
	times             = (float*) calloc(max_avg,sizeof(float)); //used for timing results
	velocities        = (float*) calloc(num_wits*wit_pts,sizeof(float)); //velocities of witnesses
	norm_velocity     = (float*) calloc(num_wits*wit_pts,sizeof(float)); //normalized velocities of witnesses
	speeds            = (float*) calloc(num_wits,sizeof(float)); //speeds of witnesses
	landmark_set      = (char*)  calloc(num_wits,sizeof(char)); //set of chosen landmarks
	cov_matrices      = (float*) calloc(num_wits*wit_pts*wit_pts,sizeof(float)); //use for d_cov distance calculation
	
	i = 0;
	j = 0;

	while(l<stop){
		
		if(l>=start){ 
			getline(&lpt,&sze,fp);

			pch = strtok(lpt," ");
			
			w=0;
			while (pch != NULL){
				
    			witnesses[i*wit_pts+w]=atof(pch);

    			pch = strtok (NULL, " ");
    			w++;
    		}
			landmark_set[l] = 'n';
			i++;
			l++;
		}
		
	}



	if(!quiet) 
		printf(" %d/%d witnesses read...",l,num_wits);
	fclose(fp);
	fp=NULL;

	printf("done\n");
	fflush(stdout);

	//calculating velocities
	float v;
	printf("Calculating vectors...");
	fflush(stdout);
	for(i=0;i<num_wits-1;i++){
		for(j=0;j<wit_pts;j++){
			v = (witnesses[i*wit_pts+wit_pts+j]-witnesses[i*wit_pts+j]);
			velocities[i*wit_pts+j] = v;
		}
	}

	//Last one's velocity is just copy of second to last one's velocity.
	for(j=0;j<wit_pts;j++){
		velocities[(num_wits-1)*wit_pts+j] = velocities[(num_wits-2)*wit_pts+j];	
	}
	
	//calculating speeds
	float sum;
	for(i=0;i<num_wits-1;i++){
		sum=0;
		for(j=0;j<wit_pts;j++){
			sum+=velocities[i*wit_pts+j]*velocities[i*wit_pts+j];
		}
		speeds[i]=sqrt(sum);
		if(speeds[i]>max_speed)
			max_speed = speeds[i];
		if(speeds[i]<min_speed)
			min_speed = speeds[i];
	}
	for(i=num_wits-2;i>=0;i--){
		if(speeds[i]==0){
			speeds[i] = speeds[i+1];
			for(j=0;j<wit_pts;j++){
				velocities[i*wit_pts+j] = velocities[(i+1)*wit_pts+j];
			}
		}
	}
	speeds[num_wits-1] = speeds[num_wits-2];


	
	for(i=0;i<num_wits-1;i++){
		if(speeds[i]==0){
			printf("for some reason %d's speed is zero\n",i);
			fflush(stdout);
		}
		for(j=0;j<wit_pts;j++){

			norm_velocity[i*wit_pts+j] = velocities[i*wit_pts+j]/speeds[i];
		}
	}
	
	for(j=0;j<wit_pts;j++){
		norm_velocity[(num_wits-1)*wit_pts+j] = norm_velocity[(num_wits-2)*wit_pts+j];
	}


	
	printf("done\n");
	fflush(stdout);

/*******************************************************************/

/***************** Calculating distance matrix ********************/

	#pragma omp parallel num_threads(num_threads) shared(euc_distance,witnesses,num_wits,wit_pts) private(i,j,k,sum)
	{
		#pragma omp for nowait schedule (runtime)
		for(i=0;i<num_wits;i++){		
			for(j=0;j<num_wits;j++){
				sum = 0;
				for(k=0;k<wit_pts;k++){

					sum+=(witnesses[i*wit_pts+k]-witnesses[j*wit_pts+k])*(witnesses[i*wit_pts+k]-witnesses[j*wit_pts+k]);
				}
				euc_distance[i*num_wits+j] = sqrt(sum);
			}
		}	
	}

	if(use_euclidean){
		printf("Calculating euclidean distances...");
		fflush(stdout);
		#pragma omp parallel num_threads(num_threads) shared(distances,witnesses,num_wits,wit_pts) private(i,j,k,sum)
		{
			#pragma omp for nowait schedule (runtime)
			for(i=0;i<num_wits;i++){		
				for(j=0;j<num_wits;j++){
					sum = 0;
//					printf("\ni, j: %d , %d \n", i, j);		        //debugging
					for(k=0;k<wit_pts;k++){
						sum+=(witnesses[i*wit_pts+k]-witnesses[j*wit_pts+k])*(witnesses[i*wit_pts+k]-witnesses[j*wit_pts+k]);
//						printf("k = %d: sum = %f \n", k, sum);		//debugging
					}
					distances[i*num_wits+j] = sqrt(sum);
				}
			}	
		}
	}
	
	else if(use_hamiltonian!=1.0){ 
		printf("Calculating hamiltonian distance...");
		fflush(stdout);
		float dhamil=0.,deuc=0.;
		if(use_hamiltonian<0){

			ham_param = use_hamiltonian*use_hamiltonian - 1;
			#pragma omp parallel num_threads(num_threads) shared(euc_distance,num_wits,witnesses,use_hamiltonian,norm_velocity,distances,wit_pts,ham_param) private(i,j,k,deuc,dhamil,sum)
			{
				#pragma omp for nowait schedule (runtime)
				for(i=0;i<num_wits;i++){
					for(j=0;j<num_wits;j++){
						deuc = euc_distance[i*num_wits+j];
						sum=0;
						
						for(k=0;k<wit_pts;k++){
							sum+=(norm_velocity[i*wit_pts+k]-norm_velocity[j*wit_pts+k])*(norm_velocity[i*wit_pts+k]-norm_velocity[j*wit_pts+k]);
						}
						if(isnan(sum)){
							printf("%d %d: one of the normalized velocities is nan\n",i,j);
							fflush(stdout);
						}
						dhamil = sqrt(sum);
						
						distances[i*num_wits+j] = sqrt(deuc*deuc+(ham_param*dhamil*dhamil));
					}
				}
			}
		}

		else{

			ham_param = use_hamiltonian*use_hamiltonian - 1;
			#pragma omp parallel num_threads(num_threads) shared(euc_distance,num_wits,witnesses,use_hamiltonian,velocities,distances,wit_pts,ham_param) private(i,j,k,deuc,dhamil,sum)
			{
				#pragma omp for nowait schedule (runtime)
				for(i=0;i<num_wits;i++){
					for(j=0;j<num_wits;j++){
						deuc = euc_distance[i*num_wits+j];
						sum=0;
						for(k=0;k<wit_pts;k++){
							sum+=(velocities[i*wit_pts+k]-velocities[j*wit_pts+k])*(velocities[i*wit_pts+k]-velocities[j*wit_pts+k]);
						}
						dhamil = sqrt(sum);
						
						distances[i*num_wits+j] = sqrt(deuc*deuc+(ham_param*dhamil*dhamil));
					}
				}
			}
		}
	}

	else if(m2_d!= 0.0){
		printf("Calculating m2_d distance..."); //m2_d
		fflush(stdout);
		float d1,d2;
		//num_wits=num_wits-m2_d;
			#pragma omp parallel num_threads(num_threads) shared(euc_distance,witnesses,num_wits,distances,m2_d) private(i,j,d1,d2)
			{
				#pragma omp for nowait schedule (runtime)
				for(i=0;i<num_wits;i++){		
					for(j=0;j<num_wits;j++){
						if(i>num_wits-m2_d-1 || j>num_wits-m2_d-1){
							distances[i*num_wits+j]=0;
						}
						else{
							d1=euc_distance[i*num_wits+j];
							d2=euc_distance[(i+m2_d)*num_wits+(j+m2_d)];
							distances[i*num_wits+j]= sqrt(d1*d1+d2*d2);
						}
					}
				}
			}
			
	}
	else if(d_cov!=0){
		
		printf("Calculating covariance distance..."); // covariance
		fflush(stdout);


		if(d_cov<0){//distance from mean of k nearest neighbors to k nearest neighbors

			d_cov*=-1;

			int neighbors[d_cov];
			float ndist[d_cov];
			
			int found = 0;
			float min = 88888;
			int min_index = -1;
			float mean[wit_pts];
			

			#pragma omp parallel num_threads(num_threads) shared(d_cov,num_wits,witnesses,euc_distance,wit_pts) private(i,j,k,l,min,min_index,neighbors,ndist,found,mean,sum)
			{
				#pragma omp for nowait schedule (runtime)
				for(i=0;i<num_wits;i++){
					found = 0;
					for(j=0;j<wit_pts;j++){
						mean[j] = 0;
					}
					//find nearest neighbors
					while(found<d_cov){
						min = 8888888;
						min_index = -1;
						for(j=0;j<num_wits;j++){
							if(euc_distance[i*num_wits+j]<min && !in_matrix(neighbors,d_cov,j) && j!=i){
								min = euc_distance[i*num_wits+j];
								min_index = j;
							}
						}
						neighbors[found] = min_index;
						ndist[found] = min;
						found++;
					}

					for(j=0;j<d_cov;j++){
						for(k=0;k<wit_pts;k++){
							mean[k] += witnesses[neighbors[j]*wit_pts+k];
						}
					}
					for(j=0;j<wit_pts;j++){
						mean[j] = mean[j]/(float)d_cov;
					}
					
					for(j=0;j<wit_pts;j++){
						for(k=0;k<wit_pts;k++){
							sum = 0;
							for(l=0;l<d_cov;l++){
								sum+=(witnesses[neighbors[l]*wit_pts+j]-mean[j])*(witnesses[neighbors[l]*wit_pts+k]-mean[k]);
							}
							cov_matrices[(i*wit_pts*wit_pts)+j*wit_pts+k] = sum/((float)d_cov-1);
						}
					}
				}
			}
		}


		else{ //instead of using mean closest witness use closest witness

			int neighbors[d_cov];
			float ndist[d_cov];
			
			int found = 0;
			float min = 88888;
			int min_index = -1;
			float closest[wit_pts];
			float sum=0;

			#pragma omp parallel num_threads(num_threads) shared(d_cov,num_wits,witnesses,euc_distance,wit_pts) private(i,j,k,l,min,min_index,neighbors,ndist,found,closest,sum)
			{
				#pragma omp for nowait schedule (runtime)
				for(i=0;i<num_wits;i++){

					found = 0;
					for(k=0;k<wit_pts;k++){
						closest[k] = 0;
					}
					for(j=0;j<d_cov;j++){
						neighbors[j] = -1;
					}

					//find nearest neighbors
					while(found<d_cov){
						min = 8888888;
						min_index = -1;
						for(j=0;j<num_wits;j++){
							if(euc_distance[i*num_wits+j]<min && !in_matrix(neighbors,d_cov,j) && j!=i){
								min = euc_distance[i*num_wits+j];
								min_index = j;
							}
						}
						neighbors[found] = min_index;
						ndist[found] = min;
						found++;
					}
					min= 888888;
					min_index = -1;
					for(j=0;j<d_cov;j++){
						if(ndist[j]<min){
							min = ndist[j];
							min_index = neighbors[j];
						}
					}

					for(j=0;j<wit_pts;j++){
						closest[j]=witnesses[min_index*wit_pts+j];	
					}
					
					for(j=0;j<wit_pts;j++){
						for(k=0;k<wit_pts;k++){
							sum = 0;
							for(l=0;l<d_cov;l++){
								sum+=(witnesses[neighbors[l]*wit_pts+j]-closest[j])*(witnesses[neighbors[l]*wit_pts+k]-closest[k]);
							}
							cov_matrices[(i*wit_pts*wit_pts)+j*wit_pts+k] = sum/((float)d_cov-1);
						}
					}

				}
			}
		}


		float c[wit_pts*wit_pts];
		
		#pragma omp parallel num_threads(num_threads) shared(distances,cov_matrices,witnesses,wit_pts,num_wits) private(i,j,k,l,c,sum)
		{
			#pragma omp for nowait schedule (runtime)
			for(i=0;i<num_wits;i++){
				for(j=0;j<num_wits;j++){
					for(k=0;k<wit_pts;k++){
						c[k] = witnesses[i*wit_pts+k]-witnesses[j*wit_pts+k];
					}
					sum=0;
					for(k=0;k<wit_pts;k++){
						for(l=0;l<wit_pts;l++){
							sum+=c[k]*c[l]*cov_matrices[(i*wit_pts*wit_pts)+k*wit_pts+l];
						}
					}
					distances[i*num_wits+j] = sum;
					if(distances[i*num_wits+j]<0)
						distances[i*num_wits+j]*=-1;
				}
			}
		}
		

	}

	else if(orientation_amplify!=1){
		float de,dot;
		printf("Calculating orientation amplify distance..."); //orientation
		fflush(stdout);
		#pragma omp parallel num_threads(num_threads) shared(num_wits,witnesses,stretch,norm_velocity,orientation_amplify,speeds,min_speed,max_speed,speed_amplify,distances,straight_VB) private(i,j,k,de,dot,sum)
		{
			#pragma omp for nowait schedule (runtime)
			for(i=0;i<num_wits;i++){
				for(j=0;j<num_wits;j++){
					sum=0;
					for(k=0;k<wit_pts;k++){
						sum+=(norm_velocity[i*wit_pts+k]-norm_velocity[j*wit_pts+k])*(norm_velocity[i*wit_pts+k]-norm_velocity[j*wit_pts+k]);
					}
					dot = sqrt(sum);
					or_param = orientation_amplify*orientation_amplify - 1;
					de = euc_distance[i*num_wits+j];
					distances[i*num_wits+j] = sqrt(de*de+(or_param*dot*de));	
				}
			}
		}
	}

/******************************************************************/

	printf("done\n");
	fflush(stdout);

	


/********************* Determing landmark set *********************/
	if(!use_est){
		landmark_set[0]='l';
		landmarks[0]=0;

		printf("Determining landmark set using MaxMin...",num_threads);
		fflush(stdout);
		float   max=-1,min=888888888,dist;
		int     max_index,landmark,min_index,l_count=1;
		float   candidate_dist[num_wits];

		while(l_count<num_landmarks){
			#pragma omp parallel num_threads(num_threads) shared(l_count,witnesses,num_wits,num_landmarks,candidate_dist,landmark_set,euc_distance) private(i,l,min,min_index,max_index,max,dist,landmark) 
			{
				#pragma omp for nowait schedule(runtime)
					for(i=0;i<num_wits;i++){
						min = 888888888;
						min_index = -1;
						if(landmark_set[i]=='n'){
							for(l=0;l<l_count;l++){
								landmark = landmarks[l];			
								dist = euc_distance[landmark*num_wits+i];
								if(dist<min){
									min = dist;
									min_index = i;
								}
							}
						}
						candidate_dist[i] = min;
					}
				#pragma omp master
				{	
					max = -8888888.;
					max_index = -1;
					for(i=0;i<num_wits;i++){
						if(candidate_dist[i]>max && landmark_set[i]=='n'){
							max = candidate_dist[i];
							max_index = i;
						}
					}
					landmarks[l_count]=max_index;
					landmark_set[max_index] = 'l';
					l_count++;
				}
				#pragma omp barrier
		  	}
		}
		printf("done\n");
		fflush(stdout);
	}

	else{

		printf("Determining landmark set using evenly spaced in time with downsample of %d...",est);
		
		fflush(stdout);

		k=0;
		for(i=0;i<num_wits;i++){
			if((i+1)%est==0 && k<num_landmarks){
				landmarks[k] = i; 
				landmark_set[i] = 'l';
				k++;
			}
		}
		printf("done\n");
	}
/******************************************************************/

/************************  GI Complex  ****************************/
	if(max_filtration_param!=0){
		printf("Computing GI Complex...");
		fflush(stdout);
		int *closest = (int*) calloc(num_wits,sizeof(int));
		
		int landmark,min_index = -1;
		float min = 888888;

		#pragma omp parallel shared(distances,closest,landmarks,num_landmarks,num_wits) private(i,j,landmark,min,min_index)
		{
			#pragma omp for nowait schedule(runtime)
			for(i=0;i<num_wits;i++){
				min_index = -1;
				min = 888888;
				for(j=0;j<num_landmarks;j++){
					landmark = landmarks[j];
					if(distances[i*num_wits+landmark]<min){
						min = distances[i*num_wits+landmark]; //SHOULD THIS BE USING EUCLIDEAN DISTANCE OF OTHER DISTANCE METRIC??
						min_index = landmark;
					}
				}
				
				closest[i] = min_index;
			}
		}



		fp = fopen("GI_edge_filtration.txt","w");
		if (fp == NULL) {
    		printf("\n\n\t\t ERROR: Failed to open output file %s!\n",wfile);
    		fflush(stdout);
    		return 1;
		}


		float max_e;

		if(max_filtration_param>0){ 
			max_e = max_filtration_param;
		}
		else{   //SOME LANDMARKS ARE ONLY CLOSEST TO THEMSELVES IN VORONOI DIAGRAM!!!!!!
			int cl;
			float K[num_wits];
			float R[num_landmarks-1];
			float min_d;
			int md,index;
			#pragma omp parallel shared(distances,landmark_set,K,closest,landmarks,num_landmarks,num_wits) private(i,j,l,landmark,min_d,R,index,md,cl)
			{
				#pragma omp for nowait schedule(runtime)
				for(i=0;i<num_wits;i++){
					cl=closest[i];
					if(landmark_set[i]=='n'){
						index = 0;
						for(l=0;l<num_landmarks;l++){
							landmark = landmarks[l];
							if(landmark!=cl){ // not looking at landmark set that witness is in
								min_d=888888;
								for(j=0;j<num_wits;j++){ //find all witnesses in that set
									if(landmark_set[j]=='n' && closest[j]==landmark){
										if(i!=j && min_d>distances[i*num_wits+j]){
											min_d = distances[i*num_wits+j];
										}
									}
								}
								R[index] = min_d;
								index++;
							}	
						}
						qsort(R, sizeof(R)/sizeof(*R), sizeof(*R), comp);
						md = -1*max_filtration_param;
						K[i] = R[md];
					}
					else{
						K[i] = 888888;
					}
					
				}


			}
			min_d=88888888;
			for(i=0;i<num_wits;i++){
				if(K[i]<min_d){
					min_d=K[i];
				}
			}
			max_e = min_d;
		}

		int *A = (int*) calloc(num_wits*num_wits,sizeof(int));
		int nd= num_divs;
		
		float thresh = 0.0;
		// assemble adjacency matrix
		int m=0; //number of total edges
		for(t = 0; t<num_divs;t++){
			thresh += (max_e/num_divs);
			for(i=0;i<num_wits;i++){
				for(j=0;j<num_wits;j++){
					if(distances[i*num_wits+j]<=thresh && closest[i]!= closest[j]){
						if(A[closest[i]*num_wits+closest[j]]==0){
							A[closest[i]*num_wits+closest[j]] = t+1;
							A[closest[j]*num_wits+closest[i]] = t+1; //ask Nikki if we need to check both distance value in matrix i.e. d[i,j] and d[j,i] for nonsymmetric distance matrices
							m+=1;	
						}
					}
				}
			}
		}



		thresh = 0.;
		if(!quiet){
			printf("Searching for triangles and edges...\n");
			printf("Number of total edges: %d\n",m);
			fflush(stdout);
		}
		int l1,l2,l3;
		bool found_tri;
		int nt,ne;
		for(t=0;t<num_divs;t++){
			thresh+=(max_e/num_divs);
			fprintf(fp,"%f:",thresh);
			nt = 0;
			ne=0;
			for(i=0;i<num_landmarks;i++){
				l1 = landmarks[i];
				for(j=0;j<num_landmarks;j++){
					l2 = landmarks[j];
					found_tri=false;
					if(l1!=l2 && A[l1*num_wits+l2]!=0 && A[l1*num_wits+l2]<=t+1){ // is edge in this filtration step
						// check for triangle
						for(k=0;k<num_landmarks;k++){
							l3 = landmarks[k];
							if(l3!=l2 && l3!=l1 && A[l1*num_wits+l3]!=0 && A[l1*num_wits+l3]<=t+1 && A[l2*num_wits+l3]!=0 && A[l2*num_wits+l3]<=t+1){
								//check if triangle is new by checking if one of the edges is new in this step, does not account for duplicate triangles within a given threshold only accross thresholds
								if(A[l1*num_wits+l3]>t || A[l2*num_wits+l3]>t || A[l1*num_wits+l2]>t){ 
									found_tri=true;
									fprintf(fp," [%d,%d,%d]",l1,l2,l3);
									nt++;
								}
							}
						}
						if(!found_tri && A[l1*num_wits+l2]>t){ //check if edge is new in this filtration step
							ne++;
							fprintf(fp," [%d,%d]",l1,l2);
						}
					}
				}
			}
			fprintf(fp,"\n");
			if(!quiet){
				printf("Threshold = %f Number of edges: %d Number of triangles: %d\n",thresh,ne,nt);
				fflush(stdout);
			}
		}
		
		printf("done\n");
		fflush(stdout);

		fclose(fp);
		free(closest);  // added by elliott 4/29
		free(A);

	}
/******************************************************************/

/************* Writing landmarks distances to file ****************/
	printf("Writing landmarks to file...");
	fflush(stdout);
	fp=fopen("landmark_outputs.txt","w");
	if (fp == NULL) {
    	printf("\n\n\t\t ERROR: Failed to open output file %s!\n",wfile);
    	fflush(stdout);
    	return 1;
	}
	fprintf(fp,"#landmark: d(l,w1), d(l,w2) ... d(l,w_n) where l refers to the landmark's occurence in the witness file, list is not sorted.\n");
	
	for(i=0;i<num_landmarks;i++){
		fprintf(fp,"%d: ",landmarks[i]);
    	
		for(j=0;j<num_wits;j++){
			if(j<num_wits-1)
			{
				
				fprintf(fp,"%f, ",distances[landmarks[i]*num_wits+j]);
			}
			else
			{
				fprintf(fp,"%f\n",distances[landmarks[i]*num_wits+j]);
			}
		}
	}
	
	fclose(fp);
	printf("done\n");
	fflush(stdout);
	
/******************************************************************/

	printf("Freeing memory...");
	fflush(stdout);
	
	//free allocated memory
	free(witnesses);
	free(times);
	free(velocities);
	free(norm_velocity);
	free(speeds);
	free(landmark_set);
	free(distances);
	free(euc_distance);
	free(landmarks);
	free(cov_matrices);



	printf("done\n");
	fflush(stdout);

	return 0;

}
int comp (const void * elem1, const void * elem2) 
{
    float f = *((float*)elem1);
    float s = *((float*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}
bool in_matrix(int *matrix, int size,int value){
	int z;
	for(z=0;z<size;z++){
		if(matrix[z]==value)
			return true;
	}
	return false;
}








