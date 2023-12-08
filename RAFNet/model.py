from blocks import *
from block_decoder import *
from block_decoder2 import *
from block_FAM import *
from block_FAM2 import *
class RegSeg(nn.Module):
    def __init__(self, name, num_classes, pretrained="", ablate_decoder=False, change_num_classes=False):
        super().__init__() 
        self.stem = ConvBnAct(3, 32, 3, 2, 1)
        #self.stem = inputlayer(3, 32) 
        body_name, decoder_name = name.split("_")
        if "eleven221418" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[1, 4]] + 2 * [[1, 8]])
        elif "eleven224488" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + 2 * [[8]])
        elif "nine2248" == body_name:
            self.body = RegSegBody(2 * [[2]] + [[4]] + [[8]] )
        elif "ten22448" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + [[8]] )
        elif "twelve22448816" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + 2*[[8]]+[[16]] )
        elif "thirteen2244881616" == body_name:
            self.body = RegSegBody(2 * [[2]] + 2 * [[4]] + 2*[[8]]+ 2*[[16]] )
        elif "twelvedown320" == body_name:
            self.body = RegSegBody2(2 * [[2]] + 2 * [[4]] + 2 * [[8]])
        elif "twelvedown321" == body_name:
            self.body = RegSegBody3(2 * [[2]] + 2 * [[4]] + 2 * [[8]])
        elif "ten1" == body_name:
            self.body = RegSegBody_1()
        elif "ten2" == body_name:
            self.body = RegSegBody_2()
        elif "ten3" == body_name:
            self.body = RegSegBody_3()
        elif "ten4" == body_name:
            self.body = RegSegBody_4()
        elif "ten5" == body_name:
            self.body = RegSegBody_5()
        elif "ten6" == body_name:
            self.body = RegSegBody_6()
        elif "fourteen1" == body_name:
            self.body = RegSegBody_7()
        elif "fourteen2" == body_name:
            self.body = RegSegBody_8()
        elif "fourteen3" == body_name:
            self.body = RegSegBody_9()
        elif "fourteen4" == body_name:
            self.body = RegSegBody_10()
        elif "eighteen1" == body_name:
            self.body = RegSegBody_11()
        elif "eighteen2" == body_name:
            self.body = RegSegBody_12()
        elif "eighteen3" == body_name:
            self.body = RegSegBody_13()
        elif "eighteen4" == body_name:
            self.body = RegSegBody_14()
        elif "regtest" == body_name:
            self.body = RegSegBody_test()
        elif "tentwo1" == body_name:
            self.body = RegSegBody_15()
        elif "CDblock" == body_name:
            self.body = CDblock()
        elif "tentwo1aspp" == body_name:
            self.body = RegSegBody_15aspp()
        elif "tentwo1320" == body_name:
            self.body = RegSegBody_152()
        elif "tentwo1short" == body_name:
            self.body = RegSegBody_153()
        elif "tentwo2" == body_name:
            self.body = RegSegBody_16()
        elif "tentwo3" == body_name:
            self.body = RegSegBody_17()
        elif "tentwo3Y" == body_name:
            self.body = RegSegBody_17Y()
        elif "tentwo3D" == body_name:
            self.body = RegSegBody_17D()
        elif "tentwo4" == body_name:
            self.body = RegSegBody_18()
        elif "tentwo5" == body_name:
            self.body = RegSegBody_19()
        elif "tentwo6" == body_name:
            self.body = RegSegBody_20()
        elif "tentwo7" == body_name:
            self.body = RegSegBody_21()
        elif "tentwo8" == body_name:
            self.body = RegSegBody_22()
        elif "tentwo9" == body_name:
            self.body = RegSegBody_23()
        elif "tentwo10" == body_name:
            self.body = RegSegBody_24()
        elif "tentwo11" == body_name:
            self.body = RegSegBody_25()
        elif "tentwo12" == body_name:
            self.body = RegSegBody_26()
        elif "tentwo13" == body_name:
            self.body = RegSegBody_27()
        elif "tentwo14" == body_name:
            self.body = RegSegBody_28()
        elif "tentwo15" == body_name:
            self.body = RegSegBody_29()
        elif "tentwo16" == body_name:
            self.body = RegSegBody_30()
        elif "testshort"==body_name:
            self.body = RegSegBody_addallshort()
        elif "testrepvgg1"==body_name:
            self.body = RegSegBody_repvgg()
        elif "mobilenetV3" ==body_name:
            self.body = MobileNetV3()
        else:
            raise NotImplementedError()
        if "decoder11" ==decoder_name:
            self.decoder = eleven_Decoder0(num_classes, self.body.channels())
        elif "decoder12" ==decoder_name:
            self.decoder = eleven_Decoder1(num_classes, self.body.channels())
        elif "decoder13" ==decoder_name:
            self.decoder = eleven_Decoder2(num_classes, self.body.channels())
        elif "LRASPP" ==decoder_name:
            self.decoder = LRASPP(num_classes, self.body.channels())
        elif "decoderdown32" == decoder_name:
            self.decoder = down32_Decoder0(num_classes, self.body.channels())
        elif "down32ASPP" == decoder_name:
            self.decoder = down32_Decoder1(num_classes, self.body.channels())
        elif "BiFPN32" == decoder_name:
            self.decoder = BiFPN([48, 128, 256, 256,19])
        elif "STDC32" == decoder_name:
            self.decoder = STDC_docker()
        elif "down32test" == decoder_name:
            self.decoder = down32_test(num_classes, self.body.channels())
        elif "EASPP" == decoder_name:
            self.decoder = eASPP_deccoder(num_classes, self.body.channels())
        elif "EASPP2" ==decoder_name:
            self.decoder = eASPP_deccoder2(num_classes, self.body.channels())
        elif "EASPP3" ==decoder_name:
            self.decoder = eASPP_deccoder3(num_classes, self.body.channels())
        elif "EASPP4" ==decoder_name:
            self.decoder = eASPP_deccoder4(num_classes, self.body.channels())
        elif "EASPP5" ==decoder_name:
            self.decoder = eASPP_deccoder5(num_classes, self.body.channels())
        elif "down32cat" == decoder_name:
            self.decoder = down32_Decoder_cat(num_classes, self.body.channels())
        elif "down32sum" == decoder_name:
            self.decoder = down32_Decoder_sum(num_classes, self.body.channels())
        elif "down3264" == decoder_name:
            self.decoder = down32_Decoder64(num_classes, self.body.channels())
        elif "down32128" == decoder_name:
            self.decoder = down32_Decoder128(num_classes, self.body.channels())
        elif "decoderproccess64" == decoder_name:
            self.decoder = down32_Decoderprocess64(num_classes, self.body.channels())
        elif "decoder1x1" == decoder_name:
            self.decoder = down32_Decoder1x1(num_classes, self.body.channels())
        elif "decoderban" == decoder_name:
            self.decoder = down32_Decoderban(num_classes, self.body.channels()) 
        elif "decoderFAM" == decoder_name:
            self.decoder = decoder_FAM(num_classes, self.body.channels())
        elif "decoderFAMfinish" == decoder_name:
            self.decoder = UperNetAlignHead(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish2" == decoder_name:
            self.decoder = UperNetAlignHead2(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish3" == decoder_name:
            self.decoder = UperNetAlignHead3(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish4" == decoder_name:
            self.decoder = UperNetAlignHead4(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv", fpn_dsn=False)
        elif "decoderFAMfinish5" == decoder_name:
            self.decoder = down32_DecoderFAM(num_classes, self.body.channels())
        elif "decoderFAMfinish6" == decoder_name:
            self.decoder = UperNetAlignHead5(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d,
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=64, conv3x3_type="conv")
        elif "decoderFAMfinish7" == decoder_name:
            self.decoder = UperNetAlignHead4(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=32, conv3x3_type="conv")
        elif "decoderFAMfinish8" == decoder_name:
            self.decoder = UperNetAlignHead6(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=128, conv3x3_type="conv")
        elif "decoderFAMfinish9" == decoder_name:
            self.decoder = UperNetAlignHead7(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=128, conv3x3_type="conv")
        elif "decoderFAMfinish10" == decoder_name:
            self.decoder = UperNetAlignHead8(inplane=256, num_class=num_classes, norm_layer=nn.BatchNorm2d, 
                                     fpn_inplanes=[48,128, 256, 256], fpn_dim=128, conv3x3_type="conv")
        elif "decoderRDD2" == decoder_name: 
            self.decoder = decoder_RDDNet_add()
        elif "decoderRDD256" == decoder_name: 
            self.decoder = decoder_RDDNet_add256()
        elif "decoderRDD3" == decoder_name: 
            self.decoder = decoder_RDDNet_addcat()
        elif "decoderRDD4" == decoder_name: 
            self.decoder = decoder_RDDNet_catadd()
        elif "decoderRDD5" == decoder_name: 
            self.decoder = decoder_RDDNet_add3x3()
        elif "decoderRDD6" == decoder_name: 
            self.decoder = decoder_RDDNet_add64()
        elif "decoderRDD7" == decoder_name: 
            self.decoder = decoder_RDDNet_add32()
        elif "decoderRDD8" == decoder_name: 
            self.decoder = decoder_RDDNet_addcommon()
        elif "decoderRDD9" == decoder_name: 
            self.decoder = decoder_RDDNet_addcommon1x1()
        elif "decoderRDD10" == decoder_name: 
            self.decoder = decoder_RDDNet_addFAM()
        elif "decoderRCD" == decoder_name: 
            self.decoder = decoder_RDDNet_add_RCD()
        
        else:
            raise NotImplementedError()
        if pretrained != "" and not ablate_decoder:
            dic = torch.load(pretrained, map_location='cpu')
            print(type(dic))
            if type(dic)==dict and "model" in dic:
                dic=dic['model']
            print(type(dic))
            if change_num_classes:
                current_model=self.state_dict()
                new_state_dict={}
                print("change_num_classes: True")
                for k in current_model:
                    if dic[k].size()==current_model[k].size():
                        new_state_dict[k]=dic[k]
                    else:
                        print(k)
                        new_state_dict[k]=current_model[k]
                self.load_state_dict(new_state_dict,strict=True)
            else:
                self.load_state_dict(dic,strict=True)
    def forward(self,x):
        input_shape=x.shape[-2:]
        x=self.stem(x)
        x=self.body(x)
        x=self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
