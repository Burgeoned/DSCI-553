import scala.collection.mutable
import java.io._
import scala.util.Try

object task1 {

  // =========================
  // global adjustables/variables
  // =========================

  //large prime for hashing
  val prime_number: Long = 2147483647L

  //array given by the pdf 
  val global_filter: Int = 69997
  val bloom_filter: Array[Int] = Array.fill[Int](global_filter)(0)

  //f(x)= (ax + b) % m or f(x) = ((ax + b) % p) % m
  //the a, b for the formula ^
  val hash_params: Array[(Int, Int)] = Array(
    (3, 7),
    (5, 11),
    (7, 13),
    (11, 17),
    (13, 19),
    (17, 23),
    (19, 29),
    (23, 31),
    (29, 37),
    (31, 41)
  )

  // =========================
  // typical generic easy utilities
  // =========================

  //old checkinputs, modified for this assignment
  def check_inputs(args: Array[String]): (String, Int, Int, String) = {
    if (args.length != 4) {
      System.err.println("Usage: task1.py <input_filename> stream_size num_of_asks <output_filename>")
      sys.exit(1)
    }

    val input_filename = args(0)
    val stream_size = Try(args(1).toInt).getOrElse {
      System.err.println("stream_size must be an integer")
      sys.exit(1); 0
    }
    val num_of_asks = Try(args(2).toInt).getOrElse {
      System.err.println("num_of_asks must be an integer")
      sys.exit(1); 0
    }
    val output_filename = args(3)

    (input_filename, stream_size, num_of_asks, output_filename)
  }

  //outputs are Time and FPR (false positive rate)
  def write_output(output_filename: String, rows: Seq[(Int, Double)]): Unit = {
    val writer = new PrintWriter(new File(output_filename))
    try {
      writer.println("Time,FPR")
      rows.foreach { case (time, fpr) =>
        writer.println(s"$time,$fpr")
      }
    } finally {
      writer.close()
    }
  }

  // =========================
  // math/bloom filter funcs
  // =========================

  //turns uid into int
  def string_to_int(uid_string: String): BigInt = {
    val hex = uid_string.getBytes("utf8").map("%02x".format(_)).mkString
    BigInt(hex, 16)
  }

  //outputs hash values 
  def myhashs(uid_string: String): Seq[Int] = {
    val x = string_to_int(uid_string)
    val indices = hash_params.map { case (a, b) =>
      val hashed = (BigInt(a) * x + BigInt(b)) mod BigInt(prime_number)
      val index = (hashed.toLong % global_filter).toInt
      index
    }
    indices.toSeq
  }

  //calculator for fpr
  def calc_fpr(fp: Long, tn: Long): Double = {
    //fpr = fp/(fp+tn)
    fp.toDouble / (fp + tn).toDouble
  }

  //takes users and checks for fp
  //fp as in the ones who have been seen (marked in bloom filter)
  def process_stream(users: Array[String], seen: mutable.Set[String]): Double = {
    var fp = 0L
    var tn = 0L

    users.foreach { uid =>
      val indices = myhashs(uid)

      //bloom filter, the hashed positions are = 1
      //no false negatives, only false positives
      val in_filter = indices.forall(i => bloom_filter(i) == 1)
      val actually_seen = seen.contains(uid)

      //in filter, but not actually seen user, = fp
      if (in_filter && !actually_seen) {
        fp += 1
      }
      //if not in filter, must be a true negative
      else if (!in_filter) {
        tn += 1
      }

      //add users into seen if they are all 1 in the filter
      indices.foreach { i =>
        bloom_filter(i) = 1
      }
      seen += uid
    }

    //fpr rate 
    calc_fpr(fp, tn)
  }

  // =========================
  // main for sauce
  // =========================

  def main(args: Array[String]): Unit = {
    val (input_filename, stream_size, num_of_asks, output_filename) = check_inputs(args)
    val bx = Blackbox()

    //stores for calcs 
    val seen_users = mutable.HashSet[String]()
    val results = mutable.ArrayBuffer[(Int, Double)]()

    //iterate through num_of_asks provided
    for (ask <- 0 until num_of_asks) {
      //call blackbox for stream based on input params 
      val users = bx.ask(input_filename, stream_size)

      //users we get vs seen ones we had 
      //process should add to seen if we see them
      val fpr = process_stream(users, seen_users)
      results.append((ask, fpr))
    }

    write_output(output_filename, results.toSeq)
  }
}
