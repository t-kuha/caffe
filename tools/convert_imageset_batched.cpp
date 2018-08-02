// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "boost/scoped_ptr.hpp"
#include "boost/thread.hpp"
#include "boost/thread/barrier.hpp"
#include "boost/atomic.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_int32(threads, 1,
    "Build DB using multiple threads.");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");


class ThreadedReader
{
public:
  typedef struct {
    int line_id;
    bool status;
    Datum datum;
  } data_entry;

  ThreadedReader(int threads, std::vector<std::pair<std::string, int> >& lines, 
    const string& encode_type, const string& root_folder, 
    int resize_height, int resize_width, bool is_color) : 
    m_thread_count(threads), m_lines(lines), 
    m_barrier_read(threads), m_barrier_fetch(2),
    m_id(0), m_thd_done(0),
    m_done(false),
    m_root_folder(root_folder),
    m_encode_type(encode_type),
    m_resize_height(resize_height), m_resize_width(resize_width),
    m_is_color(is_color)
  {
    m_data.resize(m_thread_count);
    for (int i = 0; i < threads; ++i)
      m_threads.push_back(boost::thread(boost::ref(*this)));
  }

  ~ThreadedReader() {
    for (int i = 0; i < m_thread_count; ++i)
      m_threads[i].join();
  }

  void operator()()
  {
    for (;;) {
      int my_id = m_id.fetch_add(1, boost::memory_order_acq_rel);
      const int my_base = my_id % m_thread_count;

      m_barrier_read.count_down_and_wait();

      m_data[my_base].status = false;
      m_data[my_base].line_id = my_id;

      if (my_id < m_lines.size()) {
        m_data[my_base].line_id = my_id;
        m_data[my_base].status = ReadImageToDatum(m_root_folder + m_lines[my_id].first,
          m_lines[my_id].second, m_resize_height, m_resize_width, m_is_color,
          m_encode_type, &m_data[my_base].datum);
      }

      if (m_thd_done.fetch_add(1, boost::memory_order_acq_rel) + 1 == m_thread_count) {
        m_barrier_fetch.count_down_and_wait();
        m_thd_done.store(0, boost::memory_order_release);
        if (m_id.load(boost::memory_order_relaxed) >= m_lines.size())
          m_done.store(true);
        m_barrier_fetch.count_down_and_wait();
      } 
      
      if (m_id.load(boost::memory_order_relaxed) >= m_lines.size())
        break;
    }
  }

  bool fetch_batch(std::vector<data_entry>& vec)
  {
    using std::swap;

    if (m_done.load(boost::memory_order_acquire))
      return false;

    m_barrier_fetch.count_down_and_wait();
    swap(vec, m_data);
    if (m_data.size() < m_thread_count)
      m_data.resize(m_thread_count);
    m_barrier_fetch.count_down_and_wait();

    return true;
  }

private:
  const int m_thread_count;
  std::vector<std::pair<std::string, int> >& m_lines;

  boost::barrier m_barrier_read;
  boost::barrier m_barrier_fetch;
  boost::atomic<int> m_id;
  boost::atomic<int> m_thd_done;
  boost::atomic<bool> m_done;

  std::vector<data_entry> m_data;
  std::vector<boost::thread> m_threads;

  const std::string& m_root_folder;
  const std::string& m_encode_type;
  const int m_resize_height, m_resize_width;
  const bool m_is_color;

};

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const int threads = FLAGS_threads > 1 ? FLAGS_threads : 1;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;
  std::string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  if (threads > 1) {
    ThreadedReader rdr(threads, lines, encode_type, root_folder,
      resize_height, resize_width, is_color);

    std::vector<ThreadedReader::data_entry> vec;
    while (rdr.fetch_batch(vec)) {
      for (int i = 0; i < vec.size(); ++i) {
        if (!vec[i].status)
          continue;

        if (check_size) {
          if (!data_size_initialized) {
            data_size = vec[i].datum.channels() * vec[i].datum.height() * vec[i].datum.width();
            data_size_initialized = true;
          } else {
            const std::string& data = vec[i].datum.data();
            CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
              << data.size();
          }
        }

        // sequential
        string key_str;
        stringstream o;
        o << std::setfill('0') << std::setw(8) << vec[i].line_id;
        key_str = o.str() + "_" + lines[vec[i].line_id].first;
        // Put in db
        string out;
        CHECK(vec[i].datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (++count % 1000 == 0) {
          // Commit db
          txn->Commit();
          txn.reset(db->NewTransaction());
          LOG(ERROR) << "Processed " << count << " files.";
        }

      }
    }
  } else {
    Datum datum;
    for (int line_id = 0; line_id < lines.size(); ++line_id) {
      bool status;
      std::string enc = encode_type;
      if (encoded && !enc.size()) {
        // Guess the encoding type from the file name
        string fn = lines[line_id].first;
        size_t p = fn.rfind('.');
        if (p == fn.npos)
          LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
        enc = fn.substr(p);
        std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
      }
      status = ReadImageToDatum(root_folder + lines[line_id].first,
        lines[line_id].second, resize_height, resize_width, is_color,
        enc, &datum);
      if (status == false) continue;
      if (check_size) {
        if (!data_size_initialized) {
          data_size = datum.channels() * datum.height() * datum.width();
          data_size_initialized = true;
        } else {
          const std::string& data = datum.data();
          CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();
        }
      }
      // sequential
        string key_str;
        stringstream o;
        o << std::setfill('0') << std::setw(8) << line_id;
        key_str = o.str() + "_" + lines[line_id].first.c_str();
        // Put in db
        string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);

      if (++count % 1000 == 0) {
        // Commit db
        txn->Commit();
        txn.reset(db->NewTransaction());
        LOG(ERROR) << "Processed " << count << " files.";
      }
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
