import pickle

import os
# from HTMLParser import HTMLParser

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate


def delete_file(path):
    import os
    os.remove(path)


def delete_file_no_exp(path):
    try:
        import os
        os.remove(path)
    except:
        pass


def dump_object(o, path):
    with open(path, 'wb') as f:
        pickle.dump(o, f)


def load_object(path):
    with open(path, 'rb') as f:
        var = pickle.load(f)
    return var


def send_mail_with_attach(user='shadars003@gmail.com', pwd='123456ABC', recipient='shkasta@post.bgu.ac.il',
                          subject='finish expirement', body='finish well', files=['m.txt']):
    gmail_pwd = pwd
    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = COMMASPACE.join(recipient)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(body))

    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    try:
        # SMTP_SSL Example
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo()  # optional, called by login()
        server_ssl.login(user, gmail_pwd)
        # ssl server doesn't support or need tls, so don't call server_ssl.starttls()
        server_ssl.sendmail(user, recipient, msg.as_string())
        # server_ssl.quit()
        server_ssl.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


def send_email(user='shadars003@gmail.com', pwd='123456ABC', recipient='shkasta@post.bgu.ac.il',
               subject='finish expirement', body='finish well'):
    import smtplib

    gmail_user = user
    gmail_pwd = pwd
    FROM = user
    TO = recipient if type(recipient) is list else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        # SMTP_SSL Example
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo()  # optional, called by login()
        server_ssl.login(gmail_user, gmail_pwd)
        # ssl server doesn't support or need tls, so don't call server_ssl.starttls()
        server_ssl.sendmail(FROM, TO, message)
        # server_ssl.quit()
        server_ssl.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


def flush_stout():
    import sys
    sys.stdout.flush()


def list_to_string(l):
    return "[%s]" % (" ".join(str(x) for x in l))


def is_file_exist(fname):
    import os.path
    return os.path.isfile(fname)


def get_week_day(year, month, day):
    import datetime
    datetime.datetime.today()
    d = datetime.date(year, month, day)
    return d.weekday()


def redirect_stdout(path):
    import sys
    try:
        sys.stout.close()
    except:
        pass
    sys.stdout = open(path, 'w')


def stdout_flush():
    import sys
    sys.stdout.flush()


def change_work_dir(path):
    import os
    os.chdir(path)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# class HTMLCleaner(HTMLParser):
#     container = ""
#
#     def handle_data(self, data):
#         self.container += data
#         self.container += " "
#         return self.container

def stripHTML(html):
    return html
    # p = HTMLCleaner()
    # p.feed(html)
    # result = p.container
    # for i in range(1, 10):
    #     result = result.replace("  ", " ")
    # return result
