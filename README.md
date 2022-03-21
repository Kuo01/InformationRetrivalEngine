# InformationRetrivalEngine

For backend, we use Tomcat and servlet to deal with the requests send by clients which is browser in this case. And forward the request to jsp pages. They can help present the response from server. Basically, the whole process of a user using the search engine is just to enter a query and submit. You know to send a http request. Then the tomcat and servlet will deal with the request. That’s where you can write your logic code like searching something in Lucene or printing something. Then servlet will forward the output to a jsp file which can display the output on browser.

![image](https://user-images.githubusercontent.com/86921341/159343114-63db9537-3fec-4878-bafc-a1e303ecc37c.png)
![image](https://user-images.githubusercontent.com/86921341/159343146-974c120a-0e0e-44bf-89ed-b0ae8172aa72.png)
![image](https://user-images.githubusercontent.com/86921341/159343168-567b898a-a3a9-40a7-a26a-5c041a47f6b0.png)

The following are the clustering results and Purity values of different data.

![image](https://user-images.githubusercontent.com/86921341/159376945-386cb70b-d037-4e73-b501-a419b3b1cbe5.png)

The data sets used for clustering are ICSE and VLDB. Purity values = 0.5148.

 ![image](https://user-images.githubusercontent.com/86921341/159376975-bbc7fc2a-f9bf-4070-8098-6567ac3dea23.png)
 
The data sets used for clustering are ICSE and SIGMOD. Purity values = 0.5193.

 ![image](https://user-images.githubusercontent.com/86921341/159376989-da06bd88-dd61-4b94-b08a-dc63b615b7e6.png)
 
The data sets used for clustering are SIGMOD and VLDB. Purity values = 0.5014.

 ![image](https://user-images.githubusercontent.com/86921341/159377011-e2ecfa7a-8243-4604-bb10-cd03e1ce4b8f.png)
 
The data sets used for clustering are ICSE, SIGMOD and VLDB. Purity values = 0.4439.
