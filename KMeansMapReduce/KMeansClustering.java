
import java.util.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.IOException;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;
import org.apache.hadoop.util.*;

public class KMeansClustering {
	public static String centresFile = "cluster_centres.txt";
	public static int featureLength = 0;
	public static int clusterCount = 5;
	public static class ClusterMapper extends MapReduceBase implements Mapper<LongWritable, /* Input key Type */
			Text, /* Input value Type */
			IntWritable, /* Output key Type */
			Text> /* Output value Type */
	{

		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter)
				throws IOException {
			String line = value.toString();
			String lasttoken = null;
			StringTokenizer inp = new StringTokenizer(line, " ");
			int inp_id = Integer.parseInt(inp.nextToken());
			inp.nextToken();
			int feature_counter = 0;
			ArrayList<String> input_data = new ArrayList<String>();
			while (inp.hasMoreTokens()) {
				input_data.add(inp.nextToken());
				feature_counter++;
			}
			featureLength = feature_counter;
			ArrayList<String> cluster_centers;

			Scanner s;
			cluster_centers = new ArrayList<String>();
			int N = 0;
			try {
				s = new Scanner(new File(centresFile));
				while (s.hasNext()) {
					cluster_centers.add(s.next());
					N++;
				}
				s.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			float min_dis = Float.POSITIVE_INFINITY;
			int closest_center = 0;
			int index = 0;
			int center_id = 0;
			for (String center : cluster_centers) {
				float dis = 0;
				StringTokenizer cent = new StringTokenizer(center, " ");
				while (cent.hasMoreTokens()) {
					dis += Math.abs(Float.parseFloat((cent.nextToken())) - Float.parseFloat(input_data.get(index)));
				}
				if (dis <  min_dis) {
					min_dis = dis;
					closest_center = center_id;
				}
				center_id++;
			}
			output.collect(new IntWritable(closest_center), new Text(line));
		}

	}

	// Reducer class
	public static class ClusterReducer extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {

		// Reduce function
		public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output,
				Reporter reporter) throws IOException {
			float[] val = new float[featureLength];
			float pointCounter = 0;
			StringBuilder clusterPointIds = new StringBuilder();
			while (values.hasNext()) {
				pointCounter++;
				String dataPoint = values.next().toString();
				StringTokenizer inp = new StringTokenizer(dataPoint, " "); 
				clusterPointIds.append(inp.nextToken());
				clusterPointIds.append(" ");
				inp.nextToken();
				int featureIndex = 0;
				while(inp.hasMoreTokens()) {
					val[featureIndex] += Float.parseFloat(inp.nextToken().toString());
					featureIndex++;
				}
				
			}
			StringBuilder newCenter = new StringBuilder();
			for(int i = 0; i < featureLength; i++) {
				newCenter.append(val[i]/pointCounter);
				newCenter.append(" ");
			}
			output.collect(key, new Text(newCenter.toString()));
		}
	}


	public static void main(String args[]) throws Exception {
		JobConf conf = new JobConf(KMeansClustering.class);

		conf.setJobName("kmeans");
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(IntWritable.class);
		conf.setMapperClass(ClusterMapper.class);
		conf.setCombinerClass(ClusterReducer.class);
		conf.setReducerClass(ClusterReducer.class);
		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		FileOutputFormat.setOutputPath(conf, new Path(args[1]));
		Scanner s;
		ArrayList<String> list = new ArrayList<String>();
		int N  = 0;
		try {
			s = new Scanner(new File(args[0]));
			while (s.hasNext()){
			    list.add(s.next());
			    N++;
			}
			s.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Random rand = new Random();
		int[] cntrs = new int[clusterCount];
		for(int i = 0; i < clusterCount; i++) {
			cntrs[i] = rand.nextInt(N);
		}
		FileWriter writer = new FileWriter(centresFile); 
		for(int i = 0; i < clusterCount; i++) {
		  writer.write(list.get(cntrs[i]).substring(2));
		}
		writer.close();
		Path hdfsPath = new Path(centresFile);
		DistributedCache.addCacheFile(hdfsPath.toUri(), conf);
		int i = 0;
		while(true) {
			i++;
			JobClient.runJob(conf);
			Path ofile = new Path(args[1] + "output.txt");
			FileSystem fs = FileSystem.get(new Configuration());
			BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(ofile)));
			String outLine = br.readLine();
			int clusterCount = 0;
			HashMap<Integer,String> hmap = new HashMap();
			while(outLine != null) {
				String cntr = "";
				String[] outputData = outLine.split(" ");
				for(int k = 1; k < outputData.length; k++) {
					cntr=cntr+outputData[i]+" ";
				}
				hmap.put(Integer.parseInt(outputData[0]), cntr);
				s = new Scanner(new File(centresFile));
				int diff = 0;
				int ind = 0;
				while (s.hasNext()) {
					String[] oldCntr = s.next().toString().split(" ");
					String[] newCntr = hmap.get(ind).split(" ");
					ind++;
					for(int m = 0; m < oldCntr.length; m++) {
						diff += Math.abs(Integer.parseInt(oldCntr[m]) - Integer.parseInt(newCntr[m]));
					}
				}
				if (diff > 0) {
					FileWriter wrtr = new FileWriter(centresFile); 
					for(int m = 0; m < clusterCount; m++) {
						wrtr.write(hmap.get(m));
					}
					wrtr.close();
				}
				else {
					break;
				}
				if (i > 50) {
					break;
				}
			}
		}
		
	}
}
