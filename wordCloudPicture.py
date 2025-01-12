import os
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pymysql
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("wordcloud_generator.log"),
        logging.StreamHandler()
    ]
)

# Global cache for stop words
STOP_WORDS = set()

def load_stop_words():
    """
    Load and cache stop words.
    If the stop words file does not exist or fails to read, log an error and return an empty set.
    """
    global STOP_WORDS
    if STOP_WORDS:
        return STOP_WORDS
    stop_words_path = './model/stopWords.txt'
    if not os.path.exists(stop_words_path):
        logging.error(f"Stop words file does not exist: {stop_words_path}")
        return set()
    try:
        with open(stop_words_path, encoding='utf8') as f:
            STOP_WORDS = set(line.strip() for line in f if line.strip())
        logging.info(f"Loaded {len(STOP_WORDS)} stop words")
    except Exception as e:
        logging.error(f"Failed to load stop words file: {e}")
    return STOP_WORDS

def generate_word_cloud(text, mask_path, font_path, output_path):
    """
    Generate a word cloud and save it to output_path.
    
    :param text: Processed text
    :param mask_path: Path to the mask image
    :param font_path: Path to the font file
    :param output_path: Path to save the generated word cloud image
    """
    if not os.path.exists(mask_path):
        logging.error(f"Mask image file does not exist: {mask_path}")
        return
    try:
        img = Image.open(mask_path)
        img_arr = np.array(img)
        logging.info(f"Successfully loaded mask image: {mask_path}")
    except Exception as e:
        logging.error(f"Failed to load mask image: {e}")
        return

    try:
        wc = WordCloud(
            background_color="#fff",
            mask=img_arr,
            font_path=font_path,
            max_words=2000,
            max_font_size=100,
            random_state=42,
            width=800,
            height=600
        )
        wc.generate_from_text(text)
        logging.info("Word cloud generated successfully")
    except Exception as e:
        logging.error(f"Failed to generate word cloud: {e}")
        return

    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Word cloud saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save word cloud image: {e}")

def get_db_connection_interactive():
    """
    Interactively obtain database connection parameters from the terminal.
    Press Enter to use default values.
    
    :return: pymysql.connections.Connection object
    """
    print("Please enter database connection information (press Enter to use default values):")

    host = input(" 1. Host (default: localhost): ") or "localhost"
    port_str = input(" 2. Port (default: 3306): ") or "3306"
    try:
        port = int(port_str)
    except ValueError:
        logging.error(f"Invalid port number: {port_str}")
        port = 3306

    user = input(" 3. Username (default: root): ") or "root"
    password = input(" 4. Password (default: 12345678): ") or "12345678"
    db_name = input(" 5. Database name (default: Weibo_PublicOpinion_AnalysisSystem): ") or "Weibo_PublicOpinion_AnalysisSystem"

    logging.info(f"Attempting to connect to database: {user}@{host}:{port}/{db_name}")

    try:
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db_name,
            port=port,
            charset='utf8mb4'
        )
        logging.info("Database connection successful")
        return connection
    except pymysql.MySQLError as e:
        logging.error(f"Database connection failed: {e}")
        raise

def get_img(field, table_name, target_img_src, res_img_src, connection, font_path='STHUPO.TTF'):
    """
    Retrieve text data from a specified field and table in the database,
    perform word segmentation and stop word removal, then generate a word cloud.
    
    :param field: Database field name
    :param table_name: Database table name
    :param target_img_src: Path to the mask image
    :param res_img_src: Path to save the generated word cloud image
    :param connection: Established database connection
    :param font_path: Path to the font file
    """
    try:
        with connection.cursor() as cursor:
            sql = f'SELECT {field} FROM {table_name}'
            cursor.execute(sql)
            data = cursor.fetchall()
        logging.info(f"Fetched {len(data)} records from '{table_name}' table, field '{field}'")
    except pymysql.MySQLError as e:
        logging.error(f"Database query failed: {e}")
        return

    text = ''.join(item[0] for item in data if item[0])

    # Tokenization & Stop word removal
    try:
        stop_words = load_stop_words()
        if not stop_words:
            logging.warning("Stop words set is empty, proceeding without stop word removal")
        cut_words = jieba.cut(text)
        filtered_words = [word for word in cut_words if word not in stop_words]
        final_text = ' '.join(filtered_words)
        logging.info(f"Completed tokenization and stop word removal, generated {len(filtered_words)} words")
    except Exception as e:
        logging.error(f"Text processing failed: {e}")
        return

    # Generate word cloud
    generate_word_cloud(final_text, target_img_src, font_path, res_img_src)

def main():
    """
    Main function to execute the word cloud generation process.
    """
    try:
        # Obtain database connection interactively
        connection = get_db_connection_interactive()
    except Exception:
        logging.error("Failed to establish database connection, terminating program")
        return

    try:
        # Generate word cloud as per requirements
        # Example: Generate word cloud from 'content' field in 'article' table
        get_img(
            field='content', 
            table_name='article', 
            target_img_src='./static/content.jpg', 
            res_img_src='./static/contentCloud.jpg', 
            connection=connection
        )
        print("Word cloud generation completed!")
    except Exception as e:
        logging.error(f"An error occurred during word cloud generation: {e}")
    finally:
        # Close the database connection
        try:
            connection.close()
            logging.info("Database connection closed")
        except Exception as e:
            logging.error(f"Error closing database connection: {e}")

if __name__ == '__main__':
    main()
